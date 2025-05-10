import streamlit as st
import pickle
import random
import re
import torch
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_path = "saved_mental_bert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load label encoder
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()

# Static response dictionaries
responses = {
    "Depression": [
        "You're not alone, and it's okay to ask for help. 💜",
        "Even the darkest nights end with sunrise. 🌅",
        "You deserve kindness. Would you like to share more? 💙"
    ],
    "Borderline personality disorder": [
        "You're experiencing intense emotions. You matter. 💙",
        "Your feelings are valid. You’re not alone. 🩂",
        "You are worthy of peace. 🌿"
    ],
    "Bipolar": [
        "Moods may swing, but stability is possible. 💚",
        "Managing highs and lows takes strength. 💫",
        "Would you like to share how you’ve been feeling? 🌻"
    ],
    "Anxiety": [
        "You're doing your best, and that’s enough. 💙",
        "You are stronger than your worries. 🌼",
        "You hold the power to overcome anxiety. 🌿"
    ],
    "Mentalillness": [
        "You’re stronger than you think. 💛",
        "Every effort matters. 🚤",
        "Would you like to share more about your experience? 🌸"
    ],
    "Schizophrenia": [
        "You’re not broken—you deserve care. 💜",
        "Your experiences are valid. 🌈",
        "Would you like to talk about what’s confusing? 🧩"
    ],
    "Normal": [
        "It's great to see you feeling well! 🌟",
        "Keep nurturing your well-being. 🌸",
        "You’re doing great! Keep shining. ☀️"
    ],
    "Personality disorder": [
        "You may feel misunderstood, but you're not alone. 💙",
        "You're worthy of connection. 🤝",
        "Would you like to talk more? 🌿"
    ],
    "Suicidal": [
        "Please reach out to someone you trust. ❤️",
        "Your life is precious. 🕊️",
        "Would you like to talk about what's hurting? 🥺"
    ],
    "Stress": [
        "Stress can be heavy. Have you rested today? 💚",
        "You deserve peace. 🌿",
        "One step at a time—you’re doing your best. ☁️"
    ]
}

closing_responses = {
    "Depression": random.choice([
        "💬 You deserve help and healing. 💜",
        "💬 You're not alone in this. Better days are ahead. 🌤️",
        "💬 Talking is brave. Keep reaching out. 🩂"
    ]),
    "Anxiety": random.choice([
        "💬 Gentle steps matter. 🌼",
        "💬 Inhale courage, exhale fear. You're doing well. 🌬️",
        "💬 One breath, one moment at a time. 🧘"
    ]),
    "Suicidal": random.choice([
        "💬 Please seek immediate help. 🥺",
        "💬 You matter deeply—reach out to someone you trust. 🤝",
        "💬 Stay. The world is better with you in it. 🕊️"
    ]),
    "Borderline personality disorder": random.choice([
        "💬 Compassionate care can help. 💙",
        "💬 Your emotions are valid—healing is possible. 🌊",
        "💬 You are worthy of love, stability, and peace. ⭐"
    ]),
    "Bipolar": random.choice([
        "💬 Stability is possible. 🌈",
        "💬 Your strength shines through the highs and lows. 💫",
        "💬 You're not alone—there's support for every phase. 🔀"
    ]),
    "Mentalillness": random.choice([
        "💬 You are brave. 🌻",
        "💬 Every step toward healing matters. 🚤",
        "💬 Thank you for sharing—your story deserves care. 📬"
    ]),
    "Schizophrenia": random.choice([
        "💬 Support is available. 🌟",
        "💬 You are not defined by your diagnosis. 💜",
        "💬 You deserve understanding and respect. 🎗️"
    ]),
    "Normal": random.choice([
        "💬 Keep taking care of your well-being. ✨",
        "💬 It's wonderful to hear you're feeling okay. 🌸",
        "💬 Keep spreading your light. ☀️"
    ]),
    "Personality disorder": random.choice([
        "💬 Inner peace is possible. 🌿",
        "💬 You deserve compassion and connection. 🤗",
        "💬 Healing is a journey, not a destination. 🚶‍♀️"
    ]),
    "Stress": random.choice([
        "💬 Small steps help. 🌷",
        "💬 Be kind to yourself—you’re doing your best. 💚",
        "💬 Rest and recharge—you deserve it. ☁️"
    ])
}

coping_strategies = {
    "Anxiety": [
        "🧘 Practice 4-7-8 breathing for relaxation.",
        "🎧 Listen to calming music or nature sounds.",
        "📂 Break big tasks into small steps to reduce overwhelm."
    ],
    "Depression": [
        "📓 Try journaling your thoughts for 5 minutes daily.",
        "🏃 Engage in light physical activity like a walk or stretching.",
        "🛎️ Keep a consistent sleep routine."
    ],
    "Suicidal": [
        "📞 Reach out to a crisis helpline or trusted person—you're not alone.",
        "💬 Talk to someone about how you’re feeling, even a little helps.",
        "🧩 Do something small that brings comfort, like holding a warm drink."
    ],
    "Bipolar": [
        "📆 Maintain a regular daily routine and sleep schedule.",
        "🎨 Use creative outlets like drawing or journaling to express mood shifts.",
        "🧠 Track your mood changes to better understand your cycles."
    ],
    "Schizophrenia": [
        "🔔 Set reminders for important tasks to create structure.",
        "🧩 Engage in grounding activities like listening to calming music or walking.",
        "📘 Stick to a simple routine to reduce stress."
    ],
    "Borderline personality disorder": [
        "💌 Use a feelings journal to recognize and validate your emotions.",
        "📞 Reach out to someone who makes you feel safe.",
        "🧸 Use self-soothing tools like a weighted blanket or calming scents."
    ],
    "Personality disorder": [
        "🧠 Practice self-reflection to identify emotional triggers.",
        "🤝 Consider group or one-on-one therapy to explore relationship patterns.",
        "📘 Read or listen to mental health content that promotes understanding and self-growth."
    ],
    "Mentalillness": [
        "🌱 Focus on one small, positive habit to build routine.",
        "📆 Use a daily checklist to track moods, activities, and triggers.",
        "💬 Reach out for peer support or professional help—connection matters."
    ],
    "Stress": [
        "🌿 Take a 5-minute mindful break and focus on your breathing.",
        "🎨 Do something creative like drawing or music.",
        "💧 Stay hydrated and avoid excessive caffeine."
    ]
}

follow_up_questions_by_mood = {
    "Normal": ["What’s something you're grateful for today?", "When do you feel most at peace?"],
    "Anxiety": ["Would you like to talk about your worries?", "Have you tried any grounding techniques?"],
    "Depression": ["What’s been weighing you down?", "Who can you open up to right now?"],
    "Suicidal": ["What’s been hurting the most?", "Who might support you right now?"],
    "Bipolar": ["How have your moods been recently?", "What helps you cope with the ups and downs?"],
    "Schizophrenia": ["Would you like to share your challenges?", "Is there anything grounding that helps you?"],
    "Borderline personality disorder": ["Would you like to talk about recent emotions?", "What helps you feel stable?"],
    "Personality disorder": ["Have relationships been difficult recently?", "Is there something helping you cope?"],
    "Mentalillness": ["Would you like to share your daily struggles?", "What gives you hope during hard times?"],
    "Stress": ["What’s been your biggest stress lately?", "What do you do to relax?"]
}

def predict_mood(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        label = label_encoder.inverse_transform([prediction])[0]
    return label

# Streamlit app setup
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mentoria Bot")
st.markdown("Talk to the chatbot about how you're feeling. It's safe and confidential.")
st.markdown("\n\n🔒 *Your information is not stored or shared.*")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "diagnoses" not in st.session_state:
    st.session_state.diagnoses = []

# Inputs
name = st.text_input("Hello! What's your name?", key="name")
user_input = st.text_area("How are you feeling today?", height=100)

if st.button("Get Response"):
    if name and user_input:
        predicted = predict_mood(user_input)
        chatbot_reply = random.choice(responses.get(predicted, ["I'm here to listen. 💙"]))
        follow_up = random.choice(follow_up_questions_by_mood.get(predicted, []))
        st.session_state.messages.append((user_input, chatbot_reply, follow_up, predicted))
        st.session_state.diagnoses.append(predicted)
    else:
        st.warning("Please enter both your name and how you're feeling.")

# Display conversation
for idx, (user_msg, bot_msg, question, mood) in enumerate(st.session_state.messages):
    st.markdown(f"**{name}:** {user_msg}")
    st.markdown(f"**🤖 Chatbot:** {bot_msg}")
    st.markdown(f"*Follow-up:* {question}")
    if mood == "Suicidal":
        st.markdown("\n🚨 **If you are in crisis, please contact a mental health professional or helpline in your country.**")
        st.markdown("[Find help globally here](https://www.opencounseling.com/suicide-hotlines)")
    st.markdown("---")

# Finish conversation
if st.button("Finish Conversation"):
    if st.session_state.diagnoses:
        most_common = Counter(st.session_state.diagnoses).most_common(1)[0][0]
        closing = closing_responses.get(most_common, "💬 Thank you for sharing. Take care! 💙")
        st.success(f"Hi {name}, based on our chat, you might be experiencing **{most_common}**.")
        st.info(closing)
        strategies = coping_strategies.get(most_common, [])
        if strategies:
            st.markdown("\n**Here are some strategies that might help:**")
            for strat in strategies:
                st.markdown(f"- {strat}")
    else:
        st.info("No messages to analyze yet.")

# Clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.diagnoses = []
    st.rerun()

