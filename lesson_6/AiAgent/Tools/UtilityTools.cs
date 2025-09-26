using System;
using System.ComponentModel;
using Microsoft.SemanticKernel;

public sealed class UtilityTools
{
    [KernelFunction, Description("Returns the current date and time on this computer in ISO format.")]
    public string NowIso()
        => DateTimeOffset.Now.ToString("O");

    [KernelFunction, Description("Adds two decimal numbers and returns the sum.")]
    public double Add(
        [Description("First number")] double a,
        [Description("Second number")] double b) => a + b;

    [KernelFunction, Description("Converts temperature from °C to °F.")]
    public double CelsiusToFahrenheit([Description("Temperature in degrees Celsius")] double c)
        => c * 9.0 / 5.0 + 32.0;
}