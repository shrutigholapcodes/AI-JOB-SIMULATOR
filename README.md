# AI JOB SIMULATOR

An app built built for practicing real-life workplace scenarios. Pick a field, and a role and communicate with an AI manager/client across three realistic scenarios and get a detailed evaluated performance report. 
This app is built to keep working even when the AI is unavailable. Every AI call goes through a provider chain with cooldowns, and a built-in offline engine produces follow-up questions and the full report if every provider fails or runs out of quota.

# KEY FEATURES:
1. EXCLUSIVITY : 12 ROLES IN 3 CAREER FIELD:
   
   •TECHNOLOGY - Software Developer, Data Analyst, QA Engineer, Cybersecurity Analyst
   
   •BUSINESS - Marketing, Sales, Finance, Human Resources
   
   •CREATIVE - UI/UX Designer, Graphic Designer, Content Creator, Video Editor
   
3. ADAPTIVE FOLLOW-UPS - Each scenario has 3 follow up questions which are adapted to your responses to the AI model, which keeps things pretty realistic and interesting.
   
4. DETAILED EVALUATION REPORT - An overall score out of 50 with a performance report is generated across areas like the user's problem solving skills, communication skills, decision-making and professionalism. Strength and areas for improvement are also provided along with a downloadable .txt report and a full transcript.

5. RESILIENT AI LAYER - Multiple Gemini API keys tested and trialed in order for perfection and consistency.

6. GUARDRAILS - If an answer provided by the user is too short to be evaluated, the AI reads a warning message that asks user to provide a relevant answer.

# REQUIREMENTS - 

1. Python 3.9+
   
2. Streamlit
   
3. Google-genai
   
4. Requests (only if you use Groq)

# TECHNOLOGY USED - 

1. Python - Main programming language
   
2. Streamlit - UI Framework, with session state for page navigation and st.cache_resource for shared provider state.
   
3. HTML and CSS: custom styling injected into Streamlit, using CSS variables, grid and flexbox layouts, and responsive media queries.
   
4. Google Fonts: Bebas Neue and Montserrat.
   
5. Google Gemini API: via the google-genai SDK.
    
6. Groq API: an optional fallback, called with requests against its OpenAI-compatible endpoint.
    
7. Python standard library: re, json, os, html, time and datetime.

# AREAS OF SKILL

1. LLM application development
   
•System and user prompt design.

•Structured JSON output.

•Prompt-injection defence by treating candidate text as untrusted.

•Validating and cleaning model output.

2. Resilient systems design
   
•Multi-provider, multi-key, multi-model fallback.

•Error classification with cooldowns and retirement of dead keys.

•Timeouts and attempt limits.

•Graceful degradation to a fully offline engine.

3. Natural language processing, rule-based
   
•Keyword and stem matching with regular expressions.

•Sentence extraction for evidence.

•Heuristic scoring based on coverage and answer depth.

4. Assessment design
   
•A competency rubric with five dimensions.

•Score caps for low-effort answers.

•Role-specific scoring vocabulary.

•Evidence-based feedback.

5. Content and scenario design
   
•36 workplace scenarios across 12 roles.

•"Pressure points" that drive adaptive follow-up questions.

6. Front-end and UX design
   
•A consistent visual identity.

•Report layout with progress bars and a downloadable report.

•Multi-page flow in a single-file app.

7. Security and safe output handling
   
•HTML escaping, including the $ escape that stops LaTeX rendering.

•API keys read from environment variables.

•Input validation.

8. Software engineering practices
   
•Sectioned, readable code with helper functions.

•Data-driven configuration.

•Defensive parsing and normalization of AI output against a fallback.

# CUSTOMIZING - 

Add a scenario or role: edit SCENARIOS using the S(title, opening, question, pressure_points) helper. Each role should have exactly 3 scenarios, because the progress counter assumes 9 questions. To add a new role, also add its keywords to ROLE_KEYWORDS so the offline engine can score it.

Change the look: edit THEME_CSS. The styles target Streamlit's internal data-testid attributes, which may change between Streamlit versions.

# LIMITATIONS - 

•Offline evaluation is keyword-based and can be gamed.

# DISCLAIMER

This project is for informational and educational purposes only. 

•Answer validation checks length only, not relevance, so a long but off-topic answer is accepted and left for the evaluation to score low.

•Scenarios are fixed, so repeat runs of the same role see the same openings.

•No persistence: results exist only in the session and the downloaded report.

•No automated tests yet.
