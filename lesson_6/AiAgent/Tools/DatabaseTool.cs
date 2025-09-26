using System.ComponentModel;
using System.Text;
using Microsoft.Data.Sqlite;
using Microsoft.SemanticKernel;

namespace AiAgent.Tools;

public class DatabaseTool(string connectionString)
{
    [KernelFunction,
     Description(
         "Selects data from a SQLite database using a SQL SELECT query. Returns results as a formatted string.")]
    public async Task<string> SelectAsync(
        [Description("SQL SELECT query to execute.")]
        string query)
    {
        try
        {
            await using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            await using var command = new SqliteCommand(query, connection);
            await using var reader = await command.ExecuteReaderAsync();

            var results = new List<string>();
            int fieldCount = reader.FieldCount;

            // Header
            var header = new List<string>();
            for (int i = 0; i < fieldCount; i++)
                header.Add(reader.GetName(i));
            results.Add(string.Join(" | ", header));

            // Rows
            while (await reader.ReadAsync())
            {
                var row = new List<string>();
                for (int i = 0; i < fieldCount; i++)
                    row.Add(reader.GetValue(i)?.ToString() ?? "NULL");
                results.Add(string.Join(" | ", row));
            }

            if (results.Count == 1)
                return "Query executed, no results found.";

            return string.Join(Environment.NewLine, results);
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    [KernelFunction,
     Description("Inserts a new user into the 'users' table. Requires name and age. Returns success or error message.")]
    public async Task<string> InsertUserAsync(
        [Description("User's name")] string name,
        [Description("User's age")] int age)
    {
        if (string.IsNullOrWhiteSpace(name) || age < 0)
            return "Invalid input: name must be non-empty and age must be non-negative.";

        try
        {
            await using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            await using var command = connection.CreateCommand();
            command.CommandText = "INSERT INTO users (name, age) VALUES (@name, @age);";
            command.Parameters.AddWithValue("@name", name);
            command.Parameters.AddWithValue("@age", age);

            int rows = await command.ExecuteNonQueryAsync();
            return rows > 0 ? "User inserted successfully." : "Insert failed.";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    [KernelFunction,
     Description(
         "Creates the 'users' table with columns id (INTEGER PRIMARY KEY AUTOINCREMENT), name (TEXT), and age (INTEGER).")]
    public async Task<string> CreateUsersTableAsync()
    {
        try
        {
            await using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            await using var command = connection.CreateCommand();
            command.CommandText =
                @"CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL
                );";
            await command.ExecuteNonQueryAsync();
            return "Table 'users' created (if not already exists).";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    [KernelFunction, Description("Deletes the 'users' table from the database.")]
    public async Task<string> DeleteUsersTableAsync()
    {
        try
        {
            await using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            await using var command = connection.CreateCommand();
            command.CommandText = "DROP TABLE IF EXISTS users;";
            await command.ExecuteNonQueryAsync();
            return "Table 'users' deleted (if it existed).";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    [KernelFunction, Description("Gets the schema of the entire database (all tables and their CREATE statements).")]
    public async Task<string> GetDatabaseSchemaAsync()
    {
        try
        {
            await using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            await using var command = connection.CreateCommand();
            command.CommandText =
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';";

            await using var reader = await command.ExecuteReaderAsync();
            var sb = new StringBuilder();
            while (await reader.ReadAsync())
            {
                var name = reader.GetString(0);
                var sql = reader.GetString(1);
                sb.AppendLine($"Table: {name}\n{sql}\n");
            }

            return sb.Length > 0 ? sb.ToString() : "No tables found in database.";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    [KernelFunction, Description("Gets the schema of a specific table (column names and types).")]
    public async Task<string> GetTableSchemaAsync(
        [Description("Name of the table to inspect.")]
        string tableName)
    {
        if (string.IsNullOrWhiteSpace(tableName))
            return "Table name must not be empty.";

        try
        {
            using var connection = new SqliteConnection(connectionString);
            await connection.OpenAsync();
            using var command = connection.CreateCommand();
            command.CommandText = $"PRAGMA table_info({tableName});";

            using var reader = await command.ExecuteReaderAsync();
            var sb = new StringBuilder();
            sb.AppendLine($"Schema for table '{tableName}':");
            bool hasRows = false;
            while (await reader.ReadAsync())
            {
                hasRows = true;
                var colName = reader.GetString(1);
                var colType = reader.GetString(2);
                var notNull = reader.GetInt32(3) == 1 ? "NOT NULL" : "NULL";
                var pk = reader.GetInt32(5) == 1 ? "PRIMARY KEY" : "";
                sb.AppendLine($"- {colName} ({colType}) {notNull} {pk}".Trim());
            }

            return hasRows ? sb.ToString() : $"Table '{tableName}' does not exist or has no columns.";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }
}