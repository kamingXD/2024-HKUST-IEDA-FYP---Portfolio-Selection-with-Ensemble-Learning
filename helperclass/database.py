import sqlite3
import pandas as pd


def reset_table(table_name):
    con = sqlite3.connect("./helperData/HistoryResult.db")
    cur = con.cursor()

    if table_name == "Weights":
        cur.execute("DELETE FROM Weights")
    if table_name == "Details":
        cur.execute("DELETE FROM Details")

    con.commit()
    con.close()


def append_to_weights(data):
    con = sqlite3.connect("helperData/HistoryResult.db")
    cur = con.cursor()

    cur.executemany("INSERT INTO Weights VALUES(?,?,?)", data)
    con.commit()
    con.close()


def append_to_details(data):
    con = sqlite3.connect("helperData/HistoryResult.db")
    cur = con.cursor()

    cur.executemany("INSERT INTO Details VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", data)
    con.commit()
    con.close()


def get_last_id():
    con = sqlite3.connect("helperData/HistoryResult.db")
    cur = con.cursor()
    cur.execute("SELECT RunID FROM Weights")
    rows = cur.fetchall()
    con.commit()
    con.close()

    try:
        last_id = str(int(rows[-1][0][3:])+1)
    except IndexError:
        last_id = str(1)
    '''
    if len(last_id) == 1:
        last_id = "0000" + last_id
    elif len(last_id) == 2:
        last_id = "000" + last_id
    elif len(last_id) == 3:
        last_id = "00" + last_id
    elif len(last_id) == 4:
        last_id = "0" + last_id
    '''
    return last_id


def get_table_data(table_name):
    con = sqlite3.connect("helperData/HistoryResult.db")

    if table_name == "Weights":
        df = pd.read_sql_query("SELECT * from Weights", con)
    if table_name == "Details":
        df = pd.read_sql_query("SELECT * from Details", con)

    con.commit()
    con.close()

    return df
