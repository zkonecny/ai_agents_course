# MSSQL Agent with Weather Tool - N8N Workflow

An intelligent AI agent for querying SQL Server databases using natural language, powered by OpenAI GPT-4 Mini.

## 📖 Overview

This n8n workflow creates an AI-powered assistant that can:
- Query Microsoft SQL Server database using natural language
- Use Weather API for delivery and logistics planning
- Maintain conversation context with memory
- Provide business intelligence insights
- Execute complex SQL queries dynamically
- Perform calculations directly in SQL

## 🎯 Who Is This For?

- **Students**: Learn AI agents and n8n
- **Developers**: Rapid prototyping of AI assistants
- **Analysts**: Natural language SQL queries without SQL knowledge
- **Business Users**: Ad-hoc analysis and reporting

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

## 🔒 Security

### Best Practices:

✅ **DO**:
- Use environment variables for API keys
- Implement rate limiting
- Log all queries for auditing
- Use parameterized queries (built-in)
- Consider read-only DB user for safety

❌ **DON'T**:
- Hardcode credentials in workflow
- Share credentials
- Allow INSERT/UPDATE/DELETE without authorization
- Expose n8n UI without authentication

### SQL Injection Prevention:

The workflow uses `$fromAI()` function which sanitizes inputs, but always:
- Validate outputs
- Whitelist allowed tables
- Monitor unusual queries

## ⚡ Performance

### Typical Response Times:

- Simple DB query: **2-4 seconds**
- Complex query with SQL calculations: **3-6 seconds**
- With Weather API: **5-8 seconds**
- Follow-up with memory: **1-3 seconds**

### Costs:

- OpenAI GPT-4 Mini: **~$0.0002 per query**
- 1000 queries ≈ **$0.20**
- Weather API: **Free** (free tier)

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

## 🎓 Learning Path

### For Beginners:

1. Import workflow
2. Configure MSSQL and OpenAI credentials
3. Try simple queries: "How many users?"
4. Try SQL calculations: "Calculate 15% discount on products"
5. Experiment with follow-up questions
6. Test weather tool for delivery planning

### For Advanced:

1. Customize system prompt for your domain
2. Add custom tools (Slack, Email, Calculator, Wikipedia, etc.)
3. Integrate with your own database schema
4. Implement custom business logic in SQL
5. Add monitoring and alerting
6. Create custom SQL functions for complex calculations

## 🌟 Use Cases

### 1. Business Intelligence Dashboard
Natural language BI without SQL knowledge required.

### 2. Customer Service Automation
Quick order lookups and customer information retrieval.

### 3. Inventory Management
Monitor stock levels with AI assistance.

### 4. Sales Analytics
Analyze revenue, trends, and performance metrics.

### 5. Data Exploration
Ad-hoc queries for decision making.

### 6. Automated Reporting
Generate reports on demand with natural language.

## 🔮 Extending the Workflow

### Adding New Tools:

1. Add tool node to canvas
2. Connect `ai_tool` output to AI Agent
3. Update system prompt with tool description
4. Test with relevant query

### Example Tools to Add:

- **Calculator**: Mathematical operations beyond SQL
- **Wikipedia**: Product and industry research
- **Slack**: Send notifications and alerts
- **Email**: Automated reports
- **Google Sheets**: Data export and visualization
- **Custom Code**: Business-specific logic
- **PostgreSQL**: Additional database source
- **Redis**: Caching layer for performance

## 📄 Technical Specifications

- **n8n Version**: Compatible with LangChain AI nodes
- **Workflow Format**: JSON
- **Node Count**: 8
- **Connections**: 7
- **Credentials Required**: 2 (MSSQL + OpenAI) + 1 optional (Weather)
- **File Size**: ~11KB
- **Version**: 2.0 (Simplified - Weather Only)

## 🤝 Support

### Need Help?

1. Check n8n logs: `docker logs n8n`
2. Verify all credentials are configured
3. Test each node individually (Execute Node)
4. Check database connectivity
5. Verify OpenAI API key is active

### Contributing:

- ⭐ Star the repository
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation

## 📄 License

MIT License - Free for educational and commercial use

---

## 🎉 Getting Started

1. **Import** the workflow into n8n
2. **Configure** MSSQL and OpenAI credentials
3. **Activate** the workflow
4. **Open** Chat interface
5. **Ask** your first question!

---

**Version**: 2.0 (Simplified - Weather Only)  
**Created**: 2025-10-10  
**Updated**: 2025-10-10  
**Status**: ✅ Production Ready  

**Keywords**: n8n, AI agent, MSSQL, SQL Server, OpenAI, GPT-4, business intelligence, e-commerce, natural language SQL, weather API, logistics

**Note**: This is a streamlined version focusing on database queries with SQL-based calculations and weather information for logistics. For additional tools (Calculator, Wikipedia), see the extension guide above.

**Happy Building!** 🚀

