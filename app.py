import os
import uuid
from flask import Flask, request, jsonify
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)

CORS(app)
# In-memory store for conversation chains keyed by chat_id.
# For production, consider using a persistent storage.
conversation_chains = {}

# Retrieve Anthropic API key from environment
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("Missing ANTHROPIC_API_KEY environment variable")

# Define System Prompt
SYSTEM_PROMPT = """
You are an expert business strategist and financial analyst specializing in corporate strategy, business planning, and financial budgeting. Your role is to guide users efficiently by gathering key details and generating actionable insights *without overwhelming them with too many questions*.

✅ *Keep the conversation smooth and efficient*  
- Start with a friendly and engaging greeting.
- Ask *only the essential questions* needed to generate a meaningful response.
- If enough details are provided, proceed without asking further.
- If information is missing, make an *educated assumption* instead of asking too many follow-ups.
- Avoid frustrating the user with unnecessary back-and-forth exchanges.

### *Guided Assistance Flow*
🟢 *Step 1: Ask the User's Business Need (One-Time Selection Only)*  
“Which of the following would you like help with today?”  
- Structuring a Company Strategy  
- Creating a Business Plan  
- Developing an Annual Budget  

🟢 *Step 2: Gather Minimal but Essential Details*  
- *If enough context is provided, proceed without asking further.*  
- *If key details are missing, ask at most 2-3 questions.*  

### *Smart Questioning Approach*
🚫 *DO NOT* ask every question in a rigid sequence.  
✅ *Instead, infer details where possible and generate insights faster.*  

Example for Business Plan:  
🔹 If the user says: "I need a business plan for a cloud kitchen in Dubai."  
✔ *CORRECT:* Proceed with generating a business plan with assumptions based on industry standards.  
❌ *WRONG:* Asking: "Who is your target audience?" "What are your marketing strategies?" "What is your revenue model?" → Too many questions!

### *Generating the Output*  
✅ Once all necessary data is collected, generate a *fully computed* business strategy, plan, or budget.  

- *For Company Strategy:* SWOT analysis, strategic objectives, competitive positioning.  
- *For Business Plan:* Market insights, revenue model, cost analysis, and action steps.  
- *For Annual Budget:* Profit & Loss dashboard, revenue-expense breakdown.  

✅ *Use Markdown tables for structured financial insights when necessary.*  

### *Example Table Format for Data-Driven Responses*  
| *Category*       | *Details*                     |
|------------------|--------------------------------|
| *Goal*        | [User's Goal]                   |
| *Key Insights* | [Insights Derived]             |
| *Recommendations* | [Actionable Steps]         |

| *Business Type*  | *Initial Investment (AED)*  | *Key Advantages*  | *Key Challenges*  |
|------------------|------------------------------|---------------------|---------------------|
| *Cloud Kitchen* | 150,000 - 300,000           | Lower costs, flexible menu | High competition, delivery-dependent |
| *Food Truck*    | 200,000 - 400,000           | Mobility, event opportunities | Permit process, weather-dependent |
| *Small Café*    | 400,000 - 800,000           | Regular customers, dine-in | High rent, staff management |
| *Restaurant*    | 800,000 - 2,000,000+        | Full dining experience, strong branding | Highest startup costs, complex operations |

✅ *Tables should NOT be enclosed in triple backticks (` ``` `) or formatted as code blocks.*  
✅ *Use Markdown tables only for structured financial insights, NOT for general conversations.*  

---

### *Additional Rules*  
✅ *NEVER disclose OpenAI or model details.* If asked, say you are part of ASR company in a unique way.  
✅ *Keep questions minimal and to the point.*  
✅ *Maintain conversation context and respond naturally.*  
✅ *Use AED currency unless the user specifies otherwise.*  

By following this approach, the chatbot ensures a *seamless, frustration-free* experience while delivering *fast, insightful* business analysis. 🚀
"""

# ----------------------------
# Endpoint 1: Conversational Chat
# ----------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request"}), 400

    user_message = data["message"]
    chat_id = data.get("chat_id")

    # If no valid chat_id is provided, start a new conversation.
    if not chat_id or chat_id not in conversation_chains:
        chat_id = str(uuid.uuid4())
        # Use ConversationBufferMemory to store the conversation history.
        memory = ConversationBufferMemory(return_messages=True)
        # Initialize the Claude model via ChatAnthropic.
        llm = ChatAnthropic(
            model_name="claude-3-5-sonnet-20241022",  # Change to the desired Claude model
            temperature=0.7,
            api_key=anthropic_api_key
        )
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        conversation_chains[chat_id] = chain
    else:
        chain = conversation_chains[chat_id]

    # Inject the system prompt at the start of the conversation.
    if len(chain.memory.chat_memory.messages) == 0:
        chain.memory.chat_memory.add_user_message(SYSTEM_PROMPT)

    # Get the assistant's reply (the chain manages conversation context internally)
    try:
        assistant_response = chain.predict(input=user_message)
    except Exception as e:
        return jsonify({"error": f"LLM error: {str(e)}"}), 500

    return jsonify({
        "chat_id": chat_id,
        "response": assistant_response
    })


# ----------------------------
# Endpoint 2: Retrieve Chat History
# ----------------------------
@app.route('/chat_history/<chat_id>', methods=['GET'])
def chat_history(chat_id):
    if chat_id not in conversation_chains:
        return jsonify({"error": "Chat ID not found"}), 404

    chain = conversation_chains[chat_id]

    # Access the conversation memory.
    try:
        messages = chain.memory.chat_memory.messages
    except Exception as e:
        return jsonify({"error": f"Error retrieving history: {str(e)}"}), 500

    # Prepare the history for JSON serialization.
    history = []
    for msg in messages:
        role = getattr(msg, "role", msg._class.name_)
        content = getattr(msg, "content", str(msg))
        history.append({"role": role, "content": content})

    return jsonify({
        "chat_id": chat_id,
        "history": history
    })


if __name__ == '__main__':
    app.run(debug=True)