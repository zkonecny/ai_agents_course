using AiAgent.DatabaseSetting;
using AiAgent.Tools;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var openAiApiKey = Environment.GetEnvironmentVariable("OPEN_API_KEY");
var modelId = "gpt-4o-mini";
// Example SQLite connection string (adjust as needed)
var dbConnectionString = "Data Source=mydb.sqlite;";
var dbSettings = new DatabaseSettings(dbConnectionString);
await dbSettings.PrepareDatabase();

var toolCalls = new List<string>();
IKernelBuilder builder = Kernel.CreateBuilder();
builder.AddOpenAIChatCompletion(modelId: modelId, apiKey: openAiApiKey);

// Add custom toolsets (functions)
builder.Plugins.AddFromObject(new UtilityTools(), "utils");
// Add DatabaseTool with a connection string
builder.Plugins.AddFromObject(new DatabaseTool(dbConnectionString), "db");
// Add Wikipedia search tool
builder.Plugins.AddFromObject(new WikipediaTool(), "wiki");
// Add LoggingToolFilter to track tool usage
builder.Services.AddSingleton<IFunctionInvocationFilter>(new LoggingToolFilter(toolCalls));
Kernel kernel = builder.Build();

var chat = kernel.GetRequiredService<IChatCompletionService>();

// System prompt for agent behavior
var systemPrompt =
    "You are a helpful AI agent in the console. Answer in English. " +
    "If it helps, automatically call available tools (functions). " +
    "Explain briefly and clearly.";


Console.WriteLine("=== AI Agent (Semantic Kernel + OpenAI) ===");
Console.WriteLine("Enter a question (or 'exit' to quit):");

while (true)
{
    Console.Write("> ");
    var input = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(input) || input.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    var history = new ChatHistory();
    history.AddSystemMessage(systemPrompt);
    history.AddUserMessage(input);
    toolCalls.Clear();

    var settings = new OpenAIPromptExecutionSettings
    {
        Temperature = 0.2,
        ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
    };

    var response = await chat.GetChatMessageContentAsync(history, settings, kernel);

    Console.WriteLine();
    Console.WriteLine("— Answer —");
    Console.WriteLine(response.Content?.Trim() ?? "(empty answer)");
    Console.WriteLine();

    if (toolCalls.Count > 0)
    {
        Console.WriteLine("— Used tools —");
        foreach (var t in toolCalls.Distinct())
            Console.WriteLine($"• {t}");
        Console.WriteLine();
    }
}
