=========================================================
LearnGraph AI - Universal Knowledge Engine & LLM Service
=========================================================

LearnGraph AI is an intelligent educational platform designed to provide
structured, comprehensive, and interactive learning experiences for Computer
Science, Artificial Intelligence, Algorithms, and Data Structures.

This repository contains the Universal Knowledge Engine and LLM Service Layer
that powers topic explanations, code generation, examples, quizzes, and
educational guidance.

---------------------------------------------------------
FEATURES
---------------------------------------------------------

✅ Unified LLM Service Interface
   - OpenAI Integration
   - Google Gemini Support
   - Groq AI Support
   - Offline Fallback Knowledge Engine

✅ Universal Knowledge Repository
   - Recursion
   - Binary Search
   - Dynamic Programming
   - Stacks
   - Trees
   - Graphs
   - Neural Networks
   - Extensible architecture for additional topics

✅ Intelligent Topic Matching
   - Direct concept search
   - Synonym mapping
   - Related topic detection
   - Automatic fallback explanation generation

✅ Educational Content Generation
   - Concept explanations
   - Theory and intuition
   - Time and space complexity analysis
   - Practical examples
   - Runnable Python code

✅ Assessment System
   - Multiple-choice quizzes
   - Concept-based evaluation
   - Detailed explanations
   - Learning progression support

✅ Knowledge Graph Learning Path
   - Prerequisite tracking
   - Concept dependency mapping
   - Recommended next topics
   - Guided learning progression

---------------------------------------------------------
PROJECT STRUCTURE
---------------------------------------------------------

learngraph-ai/
│
├── llm_service.py
├── .env
├── requirements.txt
├── README.txt
│
└── knowledge_engine/
    ├── recursion
    ├── binary_search
    ├── dynamic_programming
    ├── stacks
    ├── trees
    ├── graphs
    └── neural_networks

---------------------------------------------------------
ENVIRONMENT VARIABLES
---------------------------------------------------------

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_key

GROQ_API_KEY=your_groq_key

APP_MODE=live

For offline/demo mode:

APP_MODE=demo

---------------------------------------------------------
INSTALLATION
---------------------------------------------------------

1. Clone the repository

git clone https://github.com/your-username/learngraph-ai.git

2. Navigate into the project

cd learngraph-ai

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file and add provider credentials.

---------------------------------------------------------
USAGE EXAMPLE
---------------------------------------------------------

from llm_service import LLMService

service = LLMService()

result = service.universal_search("dynamic programming")

print(result["title"])
print(result["explanation"])

---------------------------------------------------------
AVAILABLE API METHODS
---------------------------------------------------------

LLMService()

Core Methods:

1. universal_search(query, subject_context=None)

Returns:
- Explanation
- Examples
- Complexity
- Prerequisites
- Unlocked Topics
- Quiz Questions

2. explain_topic(subject, topic)

Returns educational explanation for a topic.

3. give_examples(subject, topic)

Returns practical examples and code walkthroughs.

4. generate_quiz_questions(subject, topic)

Returns multiple-choice assessment questions.

5. chat_response(messages, agent_context)

Interactive conversational learning support.

---------------------------------------------------------
SUPPORTED TOPICS
---------------------------------------------------------

Algorithms
-----------
- Binary Search
- Dynamic Programming
- Recursion
- Divide and Conquer

Data Structures
---------------
- Arrays
- Stacks
- Trees
- Graphs

Artificial Intelligence
-----------------------
- Neural Networks
- Backpropagation
- Deep Learning Fundamentals

Future Expansion
----------------
- Operating Systems
- Databases
- Transformers
- Reinforcement Learning
- Computer Networks
- System Design

---------------------------------------------------------
FALLBACK KNOWLEDGE ENGINE
---------------------------------------------------------

When external AI providers are unavailable, LearnGraph AI automatically
switches to the built-in Universal Knowledge Engine.

Benefits:
- No API costs
- Fast responses
- Deterministic output
- Educational consistency
- Offline functionality

---------------------------------------------------------
DESIGN GOALS
---------------------------------------------------------

1. High-quality educational content
2. Beginner to advanced learning support
3. Minimal provider lock-in
4. Extensible knowledge architecture
5. Strong focus on learning progression
6. Interactive assessment and practice

---------------------------------------------------------
CONTRIBUTING
---------------------------------------------------------

Contributions are welcome.

Potential areas:
- New topics
- More quizzes
- Better examples
- Additional LLM providers
- Performance optimization
- Subject-specific learning paths

---------------------------------------------------------
LICENSE
---------------------------------------------------------

MIT License

Feel free to use, modify, and distribute this project according to the terms
of the license.

---------------------------------------------------------
AUTHOR
---------------------------------------------------------

LearnGraph AI

Mission:
"Transform complex computer science concepts into structured,
interactive, and intuitive learning experiences."

=========================================================
