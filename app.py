#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:02:41 2025

@author: rikcrijns
"""


import pandas as pd
import mysql.connector
import gspread
from googleapiclient.discovery import build
from numpy import nan
from flask import Flask, jsonify
from google.auth import default
from google.cloud import logging as cloud_logging
import logging
import requests
import traceback
import sys
from google.cloud import secretmanager
import json
from datetime import datetime, timedelta
from oauth2client.service_account import ServiceAccountCredentials 
import numpy as np
import os
from datetime import datetime, date
import pyodbc

app = Flask(__name__)

# --- Logging setup ---
def setup_google_cloud_logging():
    client = cloud_logging.Client()
    client.setup_logging()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    return root_logger

logger = setup_google_cloud_logging()

def log_uncaught_exceptions(ex_cls, ex, tb):
    logging.error(''.join(traceback.format_tb(tb)))
    logging.error(f'{ex_cls.__name__}: {str(ex)}')
sys.excepthook = log_uncaught_exceptions

# --- Secrets and credentials ---
def access_secret_version(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/637358609369/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(name=name)
    secret_payload = response.payload.data.decode("UTF-8")
    return secret_payload

credentials_sc_json = access_secret_version("Steamconnect-database-credentials")
credentials_sc = json.loads(credentials_sc_json)
credentials_mf_json = access_secret_version("mfs_database")
credentials_mf = json.loads(credentials_mf_json)

status_translation = {
    'pending': 'In afwachting...',
    'definitive': 'Goedgekeurd',
    'refused': 'Afval',
    'deleted': 'Verwijderd',
    'falling_out': 'Uitval',
}

def connect_to_user_db():
    host = credentials_mf["host"]
    user = credentials_mf["user"]
    database = credentials_mf["database_mfs"]
    password = credentials_mf["password"]
    port = int(credentials_mf["port"])

    
    user_db_config = {
        'user': user,        # Update with your username
        'password': password,    # Update with your password
        'host': host,    # Update with your host address
        'database': database,  # Update with your database name
        'port': port
    }
    return mysql.connector.connect(**user_db_config)


def fetch_status_hs_mf(user_db_connection, days_back):
    query = f"""
    SELECT
    	 s.order_id AS 'Result ID MF',
        s.`status` AS 'Status MF'
        
    FROM sales s
    WHERE
    	s.client_id = 21
        AND s.product_id NOT IN (24254,24255,24355,24356)

    AND
    	DATE(s.sold_on) >= DATE_SUB(CURDATE(), INTERVAL {days_back} DAY)
    """
    cursor = user_db_connection.cursor()
    cursor.execute(query)
    df = cursor.fetchall()
    mf_status_df = pd.DataFrame(df, columns=['Result ID MF', 'Status MF'])
    cursor.close()
    return mf_status_df


def process_file_hs(list_id, connection, cursor, days_back):
    # Database connection parameters
    server = '185.226.137.246'
    database = 'fonky'
    username = 'DB03_fonky'
    password = 'ee34e04331b0334ae007429198d5502a59b0c72a'
    port = 1433
    driver = '{ODBC Driver 18 for SQL Server}'
    

    
    # Create the connection string with SSL configuration
    connection_string = (
        f'DRIVER={driver};'
        f'SERVER={server},{port};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
        'TrustServerCertificate=yes;'  # Use this cautiously, only in a trusted environment
    )
    
    # Connect to the database
    conn = pyodbc.connect(connection_string, timeout=30)  # Using a 30 second timeout as specified in DBeaver
    cursor = conn.cursor()
    
    # SQL Query
    query = """
    SELECT
            c.boxPID_800 AS 'Result ID',
        c.boxPID_774 AS Werver,
        CASE
            WHEN p.tussen = '' THEN CONCAT(TRIM(p.roepnaam), ' ', TRIM(p.naam))
            ELSE CONCAT(TRIM(p.roepnaam), ' ', TRIM(p.tussen), ' ', TRIM(p.naam))
        END AS 'FCC Agent',
        c.boxPID_793 AS 'Werfdatum',
        ch.date_finishrecord AS 'Laatst gebeld',
        c.boxPID_775 AS Vestiging,
        TRIM(cr.resultcodeExport) AS 'Status',
        TRIM(cr.caption) AS 'Status betekenis',
        c.boxPID_781 AS 'Telefoonnummer',
        c.boxPID_1903 AS Woonplaats,
        CASE WHEN
                c.boxPID_1969 IS NOT NULL THEN c.boxPID_1969
                ELSE c.boxPID_794
        END AS Donatiebedrag,
        CASE WHEN 
                SUM(chs.call_count) IS NULL THEN 1
                ELSE SUM(chs.call_count)
        END AS 'Aantal keer gebeld',
        CASE WHEN
                cr.resultcode IN ('200','202', '204','205','206','207','208') THEN NULL
                ELSE c.boxPID_1914
        END AS 'Bijzonderheid',
        CASE WHEN
                cr.resultcode IN ('200','202', '204','205','206','207','208') THEN c.boxPID_1914
                ELSE NULL
        END AS 'Afval/Omzetting reden',
        CASE WHEN
                cr.resultcode IN ('100','200','201','202', '204','205','206','207','208', '300') THEN 'Ja'
                WHEN c.sys_status IN (1) THEN  'Loopt nog'
                ELSE 'Nee'
        END AS 'Bereikt',
        CASE WHEN
                cr.resultcode IN ('202', '204','205','206','207','208') AND c.sys_status NOT IN (1) THEN 'Ja'
          WHEN cr.resultcode IN ('200') AND c.sys_status NOT IN (1) AND c.boxPID_1969 IS NOT NULL AND c.boxPID_1969 < 5 THEN 'Ja'
          WHEN cr.resultcode IN ('200') AND c.sys_status NOT IN (1) AND c.boxPID_1969 IS NULL AND c.boxPID_794 < 5 THEN 'Ja'
                WHEN c.sys_status IN (1) THEN 'Loopt nog'
                ELSE 'Nee'
        END AS 'Afval',
        TRIM(crc.filenaam) as 'Voicelog'
     FROM
        C000010 c
        LEFT JOIN C000010_History ch ON c.sys_lastchpid = ch.pid
        LEFT JOIN personeel p ON ch.agentpid = p.PID
        LEFT JOIN campagnes_resultcodes cr ON c.sys_lastrc = cr.resultcode
        LEFT JOIN (
        SELECT
            cth.ctpid,
            COUNT(*) AS call_count
        FROM
            C000010_History cth
        GROUP BY
            cth.ctpid
        HAVING
            COUNT(*) > 1  -- This filters groups having more than one line
    ) chs ON c.PID = chs.ctpid
        LEFT JOIN (
                    SELECT 
                        crc.campagnehistorypid,
                        crc.campagneprojectpid,
                        MAX(crc.datumtot) AS max_datumtot
                    FROM campagnes_recordings crc
                    GROUP BY crc.campagnehistorypid, crc.campagneprojectpid
        ) crc_max ON crc_max.campagnehistorypid = ch.PID AND crc_max.campagneprojectpid = ch.projectpid
        LEFT JOIN campagnes_recordings crc ON crc.campagnehistorypid = crc_max.campagnehistorypid AND crc.datumtot = crc_max.max_datumtot AND crc.campagneprojectpid = crc_max.campagneprojectpid
    WHERE
        CAST(c.sys_importdate AS date) <= CAST(GETDATE() AS date)
        AND CONVERT(date, c.boxPID_793, 105) >= DATEADD(day, -?, GETDATE())
        AND c.boxPID_795 IS NOT NULL
        AND c.boxPID_800 IS NOT NULL
        AND cr.campagnePID = 10
    GROUP BY c.boxPID_800, c.boxPID_774,p.tussen, p.roepnaam, p.naam, c.boxPID_793, ch.date_finishrecord,c.boxPID_775,cr.resultcodeExport,
                 cr.caption,c.boxPID_781,c.boxPID_1903, c.boxPID_1969,c.boxPID_794, c.boxPID_1914, c.sys_status,crc.filenaam,cr.resultcode;
    """

    try:
        # Pass days_back as parameter for the placeholder.
        results_df = pd.read_sql_query(query, conn, params=(days_back,))
        
        # Get the status information.
        user_db_connection = connect_to_user_db()
        mf_status_df = fetch_status_hs_mf(user_db_connection, days_back)
        mf_status_df['Result ID MF'] = mf_status_df['Result ID MF'].astype(str)
        mf_status_df['Status MF'] = mf_status_df['Status MF'].astype(str)
        mf_status_df['Status MF'] = mf_status_df['Status MF'].map(status_translation)

        # Convert the SQL results "Result ID" to string for merging,
        # but also later for proper numeric sort, we convert to numeric.
        results_df['Result ID'] = pd.to_numeric(results_df['Result ID'], errors='coerce')
        results_df['Voicelog'] = results_df['Voicelog'].apply(create_voicelog_link)

        if 'Result ID' in results_df and 'Result ID MF' in mf_status_df:
            results_df['Result ID'] = results_df['Result ID'].astype(str)
            results_df = results_df.merge(mf_status_df, left_on='Result ID', right_on='Result ID MF', how='left')
            results_df.drop(columns=['Result ID MF'], inplace=True)
        
        logging.info(f"Query executed and data fetched for list_id={list_id}")
        return results_df

    except Exception as e:
        logging.error(f"Error executing query or processing results: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# --- Create voicelog link ---
def create_voicelog_link(voicelog_name):
    base_url = "https://fonky.steam.eu.com/recording/"
    if not voicelog_name:
        return ""
    try:
        date_str = voicelog_name.split('-')[0]
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        full_url = f"{base_url}{formatted_date}/{voicelog_name}.mp3"
        return full_url
    except Exception as e:
        logging.error(f"Error creating voicelog link for '{voicelog_name}': {e}")
        return ""

# --- Build Google Drive service ---
def build_drive_service():
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials, _ = default(scopes=scopes)
    if not credentials.valid:
        if credentials.requires_scopes:
            credentials = credentials.with_scopes(scopes)
    client = gspread.authorize(credentials)
    service = build('drive', 'v3', credentials=credentials)
    return service, client

# --- Make column names unique (if needed) ---
def make_column_names_unique(columns):
    seen = {}
    for i, column in enumerate(columns):
        if column in seen:
            seen[column] += 1
            columns[i] = f"{column}.{seen[column]}"
        else:
            seen[column] = 0
    return columns

# --- Fetch spreadsheet data from Google Sheets ---
def fetch_spreadsheet_data(client, sheet, days_back):
    ws = client.open("Data Goede Doelen 2024-2025").worksheet(sheet)
    data = ws.get_all_values()
    headers = data[0]
    values = data[1:]
    df = pd.DataFrame(values, columns=headers)
    df['SheetRow'] = df.index + 2  # Data rows start at row 2
    # Convert date columns to datetime (adjust format if needed)
    df['Werfdatum'] = pd.to_datetime(df['Werfdatum'], format='%d-%m-%Y', errors='coerce')
    df['Laatst gebeld'] = pd.to_datetime(df['Laatst gebeld'], format='%d-%m-%Y %H:%M', errors='coerce')
    # Filter based on days_back
    filter_date = datetime.now() - timedelta(days=days_back)
    filtered_df = df[df['Werfdatum'] >= filter_date].copy()
    # Reformat for display (if desired)
    filtered_df['Werfdatum'] = filtered_df['Werfdatum'].dt.strftime('%d-%m-%Y')
    # Convert the 'Result ID' column to numeric for later merging/sorting (if possible)
    if 'Result ID' in filtered_df.columns:
        filtered_df['Result ID'] = pd.to_numeric(filtered_df['Result ID'], errors='coerce')
    return filtered_df, df

# --- Try to parse dates (used when processing SQL results) ---
def try_parse_date(value):
    if isinstance(value, (datetime, date)):
        return pd.to_datetime(value)
    if isinstance(value, float) or isinstance(value, int):
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(int(value), unit='D')
    try:
        return pd.to_datetime(value, dayfirst=True, errors='coerce')
    except:
        return pd.NaT

def format_value_for_sheet(value, col_name=None):
    """
    Formats the value for entering into a Google Sheet.
    - If the value is null (NaN), returns an empty string.
    - For datetime values, formats using the Dutch date format.
    - Otherwise, returns the string representation.
    """
    import pandas as pd
    from datetime import datetime, date
    if pd.isnull(value):
        return ""
    if isinstance(value, (datetime, date)):
        if col_name == 'Werfdatum':
            return value.strftime('%d-%m-%Y')
        elif col_name == 'Laatst gebeld':
            return value.strftime('%d-%m-%Y %H:%M')
        else:
            return value.strftime('%d-%m-%Y %H:%M')
    return str(value)


def add_iso_columns(df):
    """
    Adds four extra columns to the DataFrame based on 'Werfdatum':
    - 'Week geworven': ISO week number
    - 'Maand geworven': Calendar month
    - 'ISO Jaar': ISO year (based on ISO week date)
    - 'ISO Kwartaal': ISO quarter (based on ISO-aligned month)
    
    If 'Werfdatum' is missing or invalid, the extra columns will be set to empty strings.
    """
    import pandas as pd
    if 'Werfdatum' in df.columns:
        # Convert 'Werfdatum' to datetime with dayfirst for European format
        df['Werfdatum'] = pd.to_datetime(df['Werfdatum'], dayfirst=True, errors='coerce')

        # ISO Week and Calendar Month
        df['Week geworven'] = df['Werfdatum'].apply(lambda x: x.isocalendar().week if pd.notnull(x) else "")
        df['Maand geworven'] = df['Werfdatum'].apply(lambda x: x.month if pd.notnull(x) else "")

        # ISO Year
        df['ISO Jaar'] = df['Werfdatum'].apply(lambda x: x.isocalendar().year if pd.notnull(x) else "")

        # ISO Quarter
        def iso_quarter(x):
            if pd.isnull(x):
                return ""
            # Adjust to ISO week Thursday
            iso_date = x - pd.to_timedelta(x.weekday(), unit='D') + pd.to_timedelta(3, unit='D')
            return (iso_date.month - 1) // 3 + 1

        df['ISO Kwartaal'] = df['Werfdatum'].apply(iso_quarter)

    else:
        df['Week geworven'] = ""
        df['Maand geworven'] = ""
        df['ISO Jaar'] = ""
        df['ISO Kwartaal'] = ""

    return df


def update_sheet(client, sheet_name, data, days_back):
    ws = client.open("Data Goede Doelen 2024-2025").worksheet(sheet_name)
    logging.info('Data Goede Doelen 2024-2025 sheet fetched for updating')
    
    # Format date columns using our helper (values are already in data and extra ISO columns are added)
    if 'Werfdatum' in data.columns:
        data['Werfdatum'] = data['Werfdatum'].apply(lambda x: format_value_for_sheet(x, 'Werfdatum'))
    if 'Laatst gebeld' in data.columns:
        data['Laatst gebeld'] = data['Laatst gebeld'].apply(lambda x: format_value_for_sheet(x, 'Laatst gebeld'))
    
    # Prepare update cells; exclude the "SheetRow" column.
    # The updated data now also includes 'ISO Week' and 'ISO Month'
    columns_to_update = [col for col in data.columns if col != 'SheetRow']
    cell_map_list = []
    
    for idx, row in data.iterrows():
        try:
            sheet_row = int(row['SheetRow'])
        except Exception as e:
            logging.error(f"Error converting SheetRow value {row.get('SheetRow')} to int: {e}")
            continue
        for col_idx, col in enumerate(columns_to_update, start=1):
            value = row[col]
            # Use helper so that NaNs become empty strings.
            formatted_value = format_value_for_sheet(value, col)
            cell_map_list.append(gspread.cell.Cell(row=sheet_row, col=col_idx, value=formatted_value))
    
    if cell_map_list:
        ws.update_cells(cell_map_list, value_input_option='USER_ENTERED')
        logging.info(f"Updated {len(cell_map_list)} cells in '{sheet_name}'.")


def append_new_leads(client, sheet_name, new_leads_df):
    ws = client.open("Data Goede Doelen 2024-2025").worksheet(sheet_name)
    logging.info('Appending new leads to sheet.')
    
    # Get the current sheet header to ensure correct column order.
    sheet_headers = ws.row_values(1)
    rows_to_append = []
    
    # Build rows that follow the header order.
    # Our new DataFrame now should include 'ISO Week' and 'ISO Month'.
    for _, row in new_leads_df.iterrows():
        new_row = [format_value_for_sheet(row[col], col) if col in row.index else "" for col in sheet_headers]
        rows_to_append.append(new_row)
    
    if rows_to_append:
        ws.append_rows(rows_to_append, value_input_option='USER_ENTERED')
        logging.info(f"Appended {len(rows_to_append)} new rows to '{sheet_name}'.")


def main():
    days_back = 28

    # Get database credentials for telforce
    json_credentials = access_secret_version("telforce_database_credentials")
    credentials = json.loads(json_credentials)
    host = credentials["host"]
    user = credentials["user"]
    database = credentials["database"]
    password = credentials["password"]
    
    service, client = build_drive_service()
    
    # Create external connection for telforce
    connection = mysql.connector.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    cursor = connection.cursor()

    sheet_to_list_id = {
        'HS': '401'
    }
    sheets = ["HS"]
    for sheet_name in sheets:
        # Fetch Google Sheet data (both filtered and full)
        filtered_sheet_df, full_sheet_df = fetch_spreadsheet_data(client, sheet_name, days_back)
        # Convert 'Result ID' to numeric for correct merging and sorting
        if 'Result ID' in filtered_sheet_df.columns:
            filtered_sheet_df['Result ID'] = pd.to_numeric(filtered_sheet_df['Result ID'], errors='coerce')

        list_id = sheet_to_list_id[sheet_name]
        print(f"Using list id {list_id}")
        
        # Fetch SQL data for this list
        results_df = process_file_hs(list_id, connection, cursor, days_back)
        print("Processing HS Sheet")
        if results_df.empty:
            logging.warning(f"No data found or error processing data for list_id={list_id}")
            return
        
        # Process results_df: convert date columns and result id
        results_df.reset_index(drop=True, inplace=True)
        results_df['Aantal keer gebeld'] = results_df['Aantal keer gebeld'].astype(int)
        results_df['Werfdatum'] = results_df['Werfdatum'].map(try_parse_date)
        results_df['Laatst gebeld'] = pd.to_datetime(results_df['Laatst gebeld'], errors='coerce')
        results_df.loc[results_df['Status'] == 'VM', 'Status betekenis'] = 'Voicemail'
        results_df.loc[((results_df['Status'] == 'GG') & (results_df['Aantal keer gebeld'] >= 6)), 'Status'] = '999'
        results_df.loc[((results_df['Status'] == 'IG') & (results_df['Aantal keer gebeld'] >= 6)), 'Status'] = '999'
        results_df.loc[(results_df['Status'] == '999'), 'Status betekenis'] = 'Max belpogingen 8'
        results_df.loc[results_df['Status'] == 'TB', 'Status betekenis'] = 'Toekomstige terugbelafspraak'
        results_df['Vestiging'] = results_df['Vestiging'].str.strip()
        results_df = results_df.drop_duplicates(subset=['Result ID']).reset_index(drop=True)
        # Convert 'Result ID' to numeric for sorting
        results_df['Result ID'] = pd.to_numeric(results_df['Result ID'], errors='coerce')
        results_df.sort_values(by=['Result ID'], inplace=True)

        # Determine which SQL results are already in the Google Sheet
        existing_ids = filtered_sheet_df['Result ID'].unique() if not filtered_sheet_df.empty else np.array([])
        
        # Split into records to update and new leads to insert
        update_df = results_df[results_df['Result ID'].isin(existing_ids)]
        new_leads_df = results_df[~results_df['Result ID'].isin(existing_ids)]
        
        # Add the ISO week and month columns to both DataFrames.
        update_df = add_iso_columns(update_df)
        new_leads_df = add_iso_columns(new_leads_df)
        
        # When updating, merge using the SheetRow from the sheet data.
        if not update_df.empty:
            merged_df = pd.merge(
                filtered_sheet_df[['Result ID', 'SheetRow']],
                update_df,
                on="Result ID",
                how="inner"
            )
            if merged_df.empty:
                print("No matching Result IDs found in the sheet to update.")
            else:
                merged_df.sort_values(by=['Result ID'], inplace=True)
                # Make sure the merged DataFrame also has our ISO columns.
                merged_df = add_iso_columns(merged_df)
                update_sheet(client, sheet_name, merged_df, days_back)
        else:
            logging.info("No existing leads to update.")
        
        # Append new leads if any
        if not new_leads_df.empty:
            # Our new leads DataFrame already has the ISO Week and ISO Month columns.
            append_new_leads(client, sheet_name, new_leads_df)
        else:
            logging.info("No new leads to append.")
            

@app.route('/')
def run_main():
    logging.info("Received request at '/' endpoint")
    try:
        main()
    except Exception as e:
        logging.error("Error occurred", exc_info=True)
        return "Internal Server Error", 500
    return "Script executed successfully."

if __name__ == '__main__':
    logging.info("Starting Flask application.")
    port = int(os.environ.get('PORT', 8080))
    app.debug = True
    app.run(host='0.0.0.0', port=port)
