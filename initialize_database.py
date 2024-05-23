import sqlite3

# Connect to SQLite database (will create it if it doesn't exist)
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Create profiles table
c.execute('''
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
)
''')

# Create descriptors table
c.execute('''
CREATE TABLE IF NOT EXISTS descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id INTEGER NOT NULL,
    descriptor BLOB NOT NULL,
    FOREIGN KEY(profile_id) REFERENCES profiles(id)
)
''')

# Create presence table
c.execute('''
CREATE TABLE IF NOT EXISTS presence (
    profile_id INTEGER PRIMARY KEY,
    is_present BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY(profile_id) REFERENCES profiles(id)
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database initialized successfully.")
