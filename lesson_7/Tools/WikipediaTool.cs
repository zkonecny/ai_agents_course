using System.ComponentModel;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;

public sealed class WikipediaTool
{
    private static readonly HttpClient _httpClient = new HttpClient();

    public WikipediaTool()
    {
        // Set User-Agent header for Wikipedia API requests
        if (!_httpClient.DefaultRequestHeaders.Contains("User-Agent"))
        {
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "AiAgentBot/1.0 (contact@example.com)");
        }
    }

    [KernelFunction,
     Description("Searches Wikipedia for the given query and returns a list of matching article titles and snippets.")]
    public async Task<string> SearchWikipediaAsync(
        [Description("Search query for Wikipedia.")]
        string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return "Query must not be empty.";

        try
        {
            string apiUrl =
                $"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={Uri.EscapeDataString(query)}&format=json";
            var response = await _httpClient.GetAsync(apiUrl);
            response.EnsureSuccessStatusCode();
            var json = await response.Content.ReadAsStringAsync();

            using var doc = JsonDocument.Parse(json);
            var searchResults = doc.RootElement.GetProperty("query").GetProperty("search");

            if (searchResults.GetArrayLength() == 0)
                return "No results found on Wikipedia.";

            var resultList = new List<string>();
            foreach (var result in searchResults.EnumerateArray())
            {
                string title = result.GetProperty("title").GetString() ?? "";
                string snippet = result.GetProperty("snippet").GetString() ?? "";
                // Remove HTML tags from snippet
                snippet = Regex.Replace(snippet, "<.*?>", "");
                resultList.Add($"{title}: {snippet}");
            }

            return string.Join(Environment.NewLine, resultList);
        }
        catch (Exception ex)
        {
            return $"WikipediaTool error: {ex.Message}";
        }
    }
}