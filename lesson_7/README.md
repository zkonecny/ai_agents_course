# AI Agent Console App (.NET 8)

This project is a console-based AI agent built with .NET 8 and [Semantic Kernel](https://github.com/microsoft/semantic-kernel). It integrates OpenAI for natural language processing and provides several tools for utility, database operations, and Wikipedia search.

## Features

- **Natural language chat** with OpenAI (GPT-4o-mini).
- **Automatic tool invocation**: The agent can call registered tools to answer user queries.
- **Extensible toolset**: Easily add new tools for custom functionality.
- **Database operations**: Query, manage, and inspect a SQLite database.
- **Wikipedia search**: Retrieve information from Wikipedia without an API key.

## Classes Overview

### `DatabaseSettings`
Located in `DatabaseSetting/DatabaseSettings.cs`  
Handles database connection string setup and preparation (e.g., creating the SQLite file if needed).

### `DatabaseTool`
Located in `Tools/DatabaseTool.cs`  
Provides database operations as Semantic Kernel functions:
- `SelectAsync`: Run SQL SELECT queries and return results.
- `CreateUsersTableAsync`: Create a `users` table (`id`, `name`, `age`).
- `DeleteUsersTableAsync`: Delete the `users` table.
- `InsertUserAsync`: Insert a new user (name, age) into the `users` table.
- `GetDatabaseSchemaAsync`: Get the schema (CREATE statements) for all tables.
- `GetTableSchemaAsync`: Get the schema for a specific table.

### `UtilityTools`
Located in `Tools/UtilityTools.cs`  
Provides general-purpose utility functions:
- `NowIso`: Returns the current date and time in ISO format.
- `Add`: Adds two decimal numbers.
- `CelsiusToFahrenheit`: Converts Celsius to Fahrenheit.

### `LoggingToolFilter`
Located in `Tools/LoggingToolFilter.cs`  
Tracks which tools are called during each chat turn. Used for logging and diagnostics.

### `WikipediaTool`
Located in `Tools/WikipediaTool.cs`  
Searches Wikipedia using the public API (no key required):
- `SearchWikipediaAsync`: Returns article titles and snippets for a search query.

## How to Use

1. **Configure OpenAI API Key**  
   Set the `OPEN_API_KEY` environment variable with your OpenAI API key.

2. **Run the App**  
   Start the console app. You will see a prompt:

3. **Ask Questions**  
Type your question. The agent will answer using OpenAI and may automatically call tools (e.g., database queries, Wikipedia search) to provide a complete answer.

4. **Tool Usage Logging**  
After each answer, the app lists which tools were used to generate the response.

## Example Interaction

=== AI Agent (Semantic Kernel + OpenAI) ===  
Enter a question (or 'exit' to quit):  
> Who is Roger Federer?

- Answer -  
Roger Federer is a Swiss former professional tennis player, born on August 8, 1981. He is widely regarded as one of the greatest tennis players of all time. Federer won 20 Grand Slam singles titles, ranking him third behind Rafael Nadal and Novak Djokovic. He has reached 31 Grand Slam finals, making him one of the most successful players in tennis history.

Federer's career is marked by his rivalries with other top players, including Rafael Nadal and Novak Djokovic, which are considered some of the greatest in the sport. He is known for his elegant playing style and has won a total of 103 ATP singles titles throughout his career.

- Used tools -  
 wiki.SearchWikipedia

## Example Tool Registration

Tools are registered in `Program.cs` like this:

You can add more tools by following this pattern.

## Requirements

- .NET 8 SDK
- OpenAI API key
- SQLite (no setup required; the app creates the database file automatically)

## Extending

To add new tools, create a class with `[KernelFunction]` methods and register it with the kernel builder.

---

**For more details, see the source code in each class file.**