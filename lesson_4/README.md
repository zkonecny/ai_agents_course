# AI Agent with database and weather tool - N8N Workflow

An intelligent AI agent for querying SQL Server databases using natural language, powered by OpenAI GPT-4 Mini.

## 📖 Overview

This n8n workflow creates an AI-powered assistant that can:
- Query Microsoft SQL Server database using natural language
- Use Weather API for delivery and logistics planning
- Maintain conversation context with memory
- Provide business intelligence insights
- Execute complex SQL queries dynamically
- Perform calculations directly in SQL

## 🚀 Quick Start

**3 steps to a working agent:**

1. **Import**: `Agent-with-database-and-tools.json` into n8n
2. **Configure**: MSSQL + OpenAI credentials
3. **Test**: "How many users do we have?"

## 🏗️ Architecture

### Components:

```
User Chat
    ↓
AI Agent (OpenAI GPT-4 Mini)
    ↓
┌───────────────┬──────────────┬─────────────┐
│   MSSQL       │   Weather    │   Memory    │
│   - Query     │   - API      │   - 10 msgs │
│   - Tables    │              │             │
│   - Schema    │              │             │
└───────────────┴──────────────┴─────────────┘
```

### Nodes (8 total):

1. **Chat Trigger**: Receives user queries via webhook
2. **AI Agent - MSSQL Assistant**: Orchestrates tool usage and reasoning
3. **OpenAI GPT-4 Mini**: Language model for understanding and generation
4. **MSSQL Query Executor**: Executes custom SQL queries (including calculations)
5. **MSSQL List Tables**: Returns all available tables
6. **MSSQL Get Table Schema**: Returns column information for specific table
7. **Weather Information**: Gets current weather data for delivery planning
8. **Conversation Memory**: Maintains context of last 10 messages

## 📊 Database Schema

### E-commerce Database

**Tables**:
- `Users` - Customer profiles (name, email, spending, etc.)
- `Products` - Product catalog (name, price, stock, rating, etc.)
- `Orders` - Order history (date, status, amount, etc.)
- `OrderItems` - Individual items in orders
- `Categories` - Product categories
- `Payments` - Payment transactions

**Example Queries**:
```sql
-- Top customers
SELECT TOP 10 FirstName, LastName, TotalSpent 
FROM Users ORDER BY TotalSpent DESC

-- Low stock
SELECT Name, StockQuantity FROM Products 
WHERE StockQuantity < 10

-- Monthly revenue
SELECT YEAR(OrderDate) as Year, 
       MONTH(OrderDate) as Month, 
       SUM(TotalAmount) as Revenue 
FROM Orders 
WHERE Status = 'Completed'
GROUP BY YEAR(OrderDate), MONTH(OrderDate)
```

## 💡 Usage Examples

### Basic Queries:

```
"How many users do we have in total?"
"Top 10 customers by spending"
"Which products have low stock?"
"Show orders from last week"
```

### Business Intelligence:

```
"What was last month's revenue?"
"Calculate average order value"
"How many new customers joined this month?"
"Which categories generate the most revenue?"
"Calculate 15% discount on all products over $100"
"Show me products with price reduced by 20%"
```

### With Weather Tool:

```
"What's the weather in Prague?" (for delivery planning)
"Check weather conditions in New York for shipping"
"Is it raining in London for delivery logistics?"
```

### Complex Queries:

```
"Find top 5 products by revenue and check their stock"
"Analyze customers who haven't ordered in 30+ days but have high spending"
"Calculate average order value per category"
```

## 🔧 Configuration

### Required Credentials:

1. **Microsoft SQL Server**
   - Host: `localhost` or IP address
   - Database: `ECommerceDB`
   - User + Password
   - Port: `1433` (default)

2. **OpenAI API**
   - API Key: `sk-...`
   - Model: GPT-4 Mini (recommended)

### Optional (but recommended):

3. **OpenWeatherMap API**
   - Free tier available
   - For weather-based delivery and logistics planning

### Settings:

- **Max Iterations**: 15 (can be reduced to 10 for simple queries)
- **Temperature**: 0.7 (lower = more consistent, higher = more creative)
- **Memory Window**: 10 messages (adjustable based on needs)

## 📈 Features

### Database Integration:
- ✅ Dynamic SQL query generation
- ✅ Schema discovery
- ✅ Table listing
- ✅ Safe parameterized queries
- ✅ SQL-based calculations (SUM, AVG, percentages, etc.)

### External Tools:
- ✅ Weather information for logistics
- ✅ Extensible (easy to add more)

### AI Capabilities:
- ✅ Natural language understanding
- ✅ Multi-step reasoning (up to 15 iterations)
- ✅ Context awareness
- ✅ Business logic integration

### Memory:
- ✅ Session-based (per user)
- ✅ 10 message buffer
- ✅ Follow-up question support

### Optimization Tips:

1. **Database**: Add indexes on frequently queried columns (Id, UserId, OrderDate)
2. **Memory**: Reduce context window for faster responses (5-7 messages)
3. **Iterations**: Lower maxIterations for simple queries (10 instead of 15)
4. **SQL**: Use aggregate functions directly in SQL for calculations
5. **Caching**: Cache frequent weather lookups

## 🐛 Troubleshooting

### "Credential not found"
- Open node → Settings → Credentials → Select/Create
- Verify credential is saved

### "Cannot connect to database"
```bash
# Check if SQL Server is running
docker ps | grep sql

# Test connection
telnet localhost 1433
sqlcmd -S localhost -U sa -P YourPassword
```

### "Agent not responding"
1. Verify OpenAI API key is valid
2. Check billing (sufficient credits)
3. Reduce maxIterations to 10
4. Try simpler query

### "Tool not called"
- Make tool descriptions more specific
- Use explicit query: "Use MSSQL and select..."
- Verify ai_tool connections between nodes

### "Slow responses"
1. Add database indexes
2. Reduce contextWindowLength to 5
3. Optimize SQL queries
4. Check network latency

## 📚 System Prompt

The AI agent is configured with a comprehensive system prompt that includes:

- **Database Schema**: Complete field descriptions for all tables
- **Query Examples**: Common SQL patterns for various scenarios
- **Tool Usage Guidelines**: When and how to use each tool
- **Best Practices**: SQL optimization, business context, error handling

The system prompt ensures the agent:
- Generates efficient SQL queries
- Provides business insights, not just raw data
- Uses appropriate tools for each task
- Maintains conversational, helpful responses