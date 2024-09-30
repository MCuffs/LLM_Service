import psycopg2
import pgvector



# Connect to your PostgreSQL database
conn = psycopg2.connect(
    host="192.168.0.20",
    port="11032",
    database="Version.1",
    user="sa",
    password="1"
)

# Example: Use the connection
cur = conn.cursor()

create_table_command = """
CREATE TABLE vectors (
    id bigserial primary key,
    embedding vector(3)  -- specify the dimension of the vector
);
"""
# Execute the SQL command
cur.execute(create_table_command)
# Commit the transaction
conn.commit()
