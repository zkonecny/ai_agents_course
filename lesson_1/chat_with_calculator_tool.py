import os
import json
import requests
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Tool functions
def calculator_tool(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Chyba: {e}"


# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator_tool",
            "description": "Spočítá matematický výraz",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Matematický výraz k vyhodnocení"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

available_functions = {
    "calculator_tool": calculator_tool
}


# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,   # Custom tools
        tool_choice="auto"  # Allow AI to decide if a tool should be called
    )

    response_message = response.choices[0].message
    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)
        print(function_response)

        # Append tool call and result back into conversation
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return response.choices[0].message


# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Spočítej 12 * (3 + 4)"}
    #{"role": "user", "content": "Kdo je v Česku prezident?"}
]

response = get_completion_from_messages(messages)

print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
content = getattr(response, "content", None)
if content:
    print(content)
else:
    print("No text content in the response")