using Microsoft.Data.Sqlite;

namespace AiAgent.DatabaseSetting;

public class DatabaseSettings
{
    private readonly string _connectionString;

    public DatabaseSettings(string connectionString)
    {
        _connectionString = connectionString;
    }

    public async Task PrepareDatabase()
    {
        await CreateUsersTableAsync();
        await InsertSampleUsersAsync();
    }
    
    public async Task<string> CreateUsersTableAsync()
    {
        try
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();
            using var command = connection.CreateCommand();
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

    public async Task<string> DeleteUsersTableAsync()
    {
        try
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();
            using var command = connection.CreateCommand();
            command.CommandText = "DROP TABLE IF EXISTS users;";
            await command.ExecuteNonQueryAsync();
            return "Table 'users' deleted (if it existed).";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }

    public async Task<string> InsertSampleUsersAsync()
    {
        var users = new[]
        {
            new { Name = "Alice", Age = 25 },
            new { Name = "Bob", Age = 30 },
            new { Name = "Charlie", Age = 22 },
            new { Name = "Diana", Age = 28 },
            new { Name = "Eve", Age = 35 },
            new { Name = "Frank", Age = 27 },
            new { Name = "Grace", Age = 24 },
            new { Name = "Heidi", Age = 29 },
            new { Name = "Ivan", Age = 31 },
            new { Name = "Judy", Age = 26 }
        };

        try
        {
            using var connection = new SqliteConnection(_connectionString);
            await connection.OpenAsync();

            int inserted = 0;
            foreach (var user in users)
            {
                using var command = connection.CreateCommand();
                command.CommandText = "INSERT INTO users (name, age) VALUES (@name, @age);";
                command.Parameters.AddWithValue("@name", user.Name);
                command.Parameters.AddWithValue("@age", user.Age);
                inserted += await command.ExecuteNonQueryAsync();
            }

            return $"Inserted {inserted} sample users into 'users' table.";
        }
        catch (Exception ex)
        {
            return $"Database error: {ex.Message}";
        }
    }
}