
import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import plotly.express as px
from datetime import datetime

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Skincare Customer Feedback",
    page_icon="🧴",
    layout="wide"
)

st.title("Online Skincare Store")
st.caption("Customer Feedback & Sentiment Analysis")

# --------------------------------------------------
# 2. Groq API Configuration
# --------------------------------------------------

try:
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
except Exception:
    st.error(
        "Groq API key is missing. Please configure "
        "GROQ_API_KEY in Streamlit Secrets."
    )
    st.stop()

# --------------------------------------------------
# 3. Labels and Classification Prompt
# --------------------------------------------------

labels = [
    "Positive",
    "Negative",
    "Mixed",
    "Neutral / Unclear"
]

system_prompt = """
You are a customer feedback classification assistant
for an online skincare store.

Classify each customer comment into exactly ONE label:

1. Positive:
Customer expresses clear satisfaction or a positive experience.

2. Negative:
Customer expresses dissatisfaction, disappointment, or a complaint.

3. Mixed:
Customer expresses both positive and negative experiences.

4. Neutral / Unclear:
Customer provides information or asks a question without
a clear positive or negative opinion.

Classification rules:
- Each comment must receive exactly one label.
- If a comment contains both praise and a complaint, use Mixed.
- If there is a clear complaint without meaningful praise, use Negative.
- If there is no clear positive or negative opinion, use Neutral / Unclear.
- Do not invent information.
- Skin irritation, allergic reactions, or other safety concerns
  must be flagged for manual review.

Return ONLY a valid JSON object:
{
    "label": "Negative",
    "safety_concern": true,
    "reason": "The customer reports skin irritation."
}

The label must be one of the four allowed labels.
safety_concern must be true or false.
"""

# --------------------------------------------------
# 4. Database Setup
# --------------------------------------------------

DB_NAME = "customer_feedback.db"


def initialize_database():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rating INTEGER,
            comment TEXT,
            label TEXT,
            safety_concern BOOLEAN,
            manual_review BOOLEAN,
            review_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


initialize_database()

# --------------------------------------------------
# 5. Classify Customer Comment
# --------------------------------------------------

def classify_comment(comment):

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": comment}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(
        response.choices[0].message.content
    )

    if result.get("label") not in labels:
        raise ValueError("Invalid classification label.")

    if not isinstance(result.get("safety_concern"), bool):
        raise ValueError("Invalid safety concern result.")

    if not isinstance(result.get("reason"), str):
        raise ValueError("Invalid classification reason.")

    return result


# --------------------------------------------------
# 6. Save Customer Feedback
# --------------------------------------------------

def save_feedback(rating, comment):

    label = None
    safety_concern = False
    review_reason = None

    if comment and comment.strip():

        result = classify_comment(comment)

        label = result["label"]
        safety_concern = result["safety_concern"]
        review_reason = result["reason"]

    # Check rating and sentiment mismatch

    rating_mismatch = False

    if rating is not None and label is not None:

        if rating <= 2 and label == "Positive":
            rating_mismatch = True

        elif rating >= 4 and label == "Negative":
            rating_mismatch = True

    manual_review = safety_concern or rating_mismatch

    if rating_mismatch:
        mismatch_reason = (
            "Customer rating and comment sentiment do not match."
        )

        review_reason = (
            f"{review_reason} {mismatch_reason}"
            if review_reason
            else mismatch_reason
        )

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO feedback (
            rating,
            comment,
            label,
            safety_concern,
            manual_review,
            review_reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        rating,
        comment.strip() if comment else None,
        label,
        safety_concern,
        manual_review,
        review_reason
    ))

    conn.commit()
    conn.close()

    return label, manual_review, review_reason


# --------------------------------------------------
# 7. Load Feedback Data
# --------------------------------------------------

def load_feedback():

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM feedback ORDER BY created_at DESC",
        conn
    )

    conn.close()

    return df


# --------------------------------------------------
# 8. Sidebar Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Customer Feedback", "Owner Dashboard"]
)


# ==================================================
# PAGE 1: CUSTOMER FEEDBACK
# ==================================================

if page == "Customer Feedback":

    st.header("We'd Love to Hear From You!")

    st.write(
        "Your feedback helps us improve our skincare products "
        "and shopping experience."
    )

    with st.form("customer_feedback_form"):

        rating = st.selectbox(
            "How would you rate your experience?",
            options=[None, 1, 2, 3, 4, 5],
            format_func=lambda x: (
                "Not provided" if x is None else f"{x} ⭐"
            )
        )

        comment = st.text_area(
            "Share your experience",
            placeholder=(
                "Tell us about your experience with our products..."
            ),
            height=150
        )

        submitted = st.form_submit_button(
            "Submit Feedback",
            type="primary"
        )

        if submitted:

            if rating is None and not comment.strip():

                st.warning(
                    "Please provide a star rating or a written comment."
                )

            else:

                try:

                    with st.spinner(
                        "Processing your feedback..."
                    ):

                        label, manual_review, reason = save_feedback(
                            rating,
                            comment
                        )

                    st.success(
                        "Thank you! Your feedback has been submitted."
                    )

                    if label:
                        st.write(f"Feedback category: **{label}**")

                    if manual_review:
                        st.info(
                            "Your feedback has been flagged for additional review."
                        )

                except Exception as e:

                    st.error(
                        f"Unable to process your feedback: {str(e)}"
                    )


# ==================================================
# PAGE 2: OWNER DASHBOARD
# ==================================================

elif page == "Owner Dashboard":

    st.header("Business Owner Dashboard")

    st.write(
        "Monitor customer satisfaction and identify "
        "feedback that requires attention."
    )

    df = load_feedback()

    if df.empty:

        st.info(
            "No customer feedback has been submitted yet. "
            "Submit some feedback to populate the dashboard."
        )

    else:

        # ------------------------------------------
        # Key Metrics
        # ------------------------------------------

        total_feedback = len(df)

        average_rating = df["rating"].mean()

        manual_review_count = int(
            df["manual_review"].sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Feedback",
            total_feedback
        )

        col2.metric(
            "Average Star Rating",
            (
                f"{average_rating:.2f} / 5"
                if pd.notna(average_rating)
                else "N/A"
            )
        )

        col3.metric(
            "Needs Manual Review",
            manual_review_count
        )

        st.divider()

        # ------------------------------------------
        # Graph 1: Feedback by Label
        # ------------------------------------------

        st.subheader("Customer Feedback by Sentiment")

        labeled_df = df.dropna(subset=["label"])

        if not labeled_df.empty:

            label_counts = (
                labeled_df["label"]
                .value_counts()
                .reindex(labels, fill_value=0)
                .reset_index()
            )

            label_counts.columns = [
                "Sentiment",
                "Number of Comments"
            ]

            fig1 = px.bar(
                label_counts,
                x="Sentiment",
                y="Number of Comments",
                title="Number of Comments by Sentiment",
                color="Sentiment"
            )

            st.plotly_chart(
                fig1,
                width="stretch"
            )

        else:

            st.info("No written comments available for sentiment analysis.")

        # ------------------------------------------
        # Graph 2: Average Rating by Label
        # ------------------------------------------

        st.subheader("Average Star Rating by Sentiment")

        rating_df = df.dropna(
            subset=["label", "rating"]
        )

        if not rating_df.empty:

            avg_rating_df = (
                rating_df.groupby("label")["rating"]
                .mean()
                .reindex(labels)
                .dropna()
                .reset_index()
            )

            avg_rating_df.columns = [
                "Sentiment",
                "Average Rating"
            ]

            fig2 = px.bar(
                avg_rating_df,
                x="Sentiment",
                y="Average Rating",
                title="Average Star Rating by Sentiment",
                color="Sentiment"
            )

            fig2.update_yaxes(
                range=[0, 5]
            )

            st.plotly_chart(
                fig2,
                width="stretch"
            )

        else:

            st.info(
                "Not enough rated comments to calculate average ratings."
            )

        st.divider()

        # ------------------------------------------
        # Manual Review
        # ------------------------------------------

        st.subheader("Comments Requiring Manual Review")

        review_df = df[
            df["manual_review"] == 1
        ]

        if review_df.empty:

            st.success(
                "There are currently no comments requiring manual review."
            )

        else:

            st.warning(
                f"{len(review_df)} comments require manual review."
            )

            for _, row in review_df.iterrows():

                with st.expander(
                    f"Feedback #{row['id']} | "
                    f"Rating: {row['rating'] if pd.notna(row['rating']) else 'Not provided'} ⭐"
                ):

                    st.write("**Full Customer Comment:**")

                    st.write(
                        row["comment"]
                        if row["comment"]
                        else "No written comment."
                    )

                    st.write(
                        f"**LLM Label:** {row['label'] or 'Not available'}"
                    )

                    st.write(
                        f"**Safety Concern:** "
                        f"{'Yes' if row['safety_concern'] else 'No'}"
                    )

                    st.write("**Reason for Review:**")

                    st.write(
                        row["review_reason"] or "No reason provided."
                    )

                    st.caption(
                        f"Submitted: {row['created_at']}"
                    )

        st.divider()

        # ------------------------------------------
        # All Feedback Records
        # ------------------------------------------

        with st.expander("View All Customer Feedback"):

            st.dataframe(
                df,
                hide_index=True,
                width="stretch"
            )
