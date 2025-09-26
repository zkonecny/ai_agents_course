using Microsoft.SemanticKernel;

public sealed class LoggingToolFilter : IFunctionInvocationFilter
{
    private readonly IList<string> _sink;
    public LoggingToolFilter(IList<string> sink) => _sink = sink;

    public async Task OnFunctionInvocationAsync(FunctionInvocationContext context,
        Func<FunctionInvocationContext, Task> next)
    {
        var name = $"{context.Function.PluginName}.{context.Function.Name}";
        _sink.Add(name);

        await next(context);
    }
}