import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="WhatsApp Chat Analyzer Pro",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #25D366, #128C7E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #25D366;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize URL extractor
extractor = URLExtract()


def preprocess(data):
    """Enhanced preprocessing with better error handling and data validation"""
    try:
        # Multiple patterns to handle different WhatsApp export formats
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[AP]M\s-\s',  # MM/DD/YY, HH:MM AM/PM -
            r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',  # MM/DD/YY, HH:MM -
            r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s[AP]M\]',  # [MM/DD/YY, HH:MM:SS AM/PM]
        ]

        # Try different patterns
        for i, pattern in enumerate(patterns):
            messages = re.split(pattern, data)
            dates = re.findall(pattern, data)

            if len(messages) > 1 and len(dates) > 0:
                messages = messages[1:]  # Remove first empty element
                break

        if len(messages) <= 1:
            st.error("Could not parse the chat file. Please ensure it's a valid WhatsApp export.")
            return None

        # Clean dates
        cleaned_dates = []
        for d in dates:
            d = d.replace(' - ', '').replace('[', '').replace(']', '')
            cleaned_dates.append(d)

        df = pd.DataFrame({'user_message': messages, "message_date": cleaned_dates})

        # Try different datetime formats
        date_formats = [
            '%m/%d/%y, %I:%M %p',
            '%m/%d/%Y, %I:%M %p',
            '%m/%d/%y, %H:%M',
            '%m/%d/%Y, %H:%M',
            '%m/%d/%y, %I:%M:%S %p',
            '%m/%d/%Y, %I:%M:%S %p'
        ]

        for fmt in date_formats:
            try:
                df['message_date'] = pd.to_datetime(df['message_date'], format=fmt)
                break
            except:
                continue

        if df['message_date'].dtype == 'object':
            st.error("Could not parse date format. Please check your chat export format.")
            return None

        df.rename(columns={'message_date': 'date'}, inplace=True)

        # Separate users and messages with improved regex
        users = []
        messages = []

        for message in df['user_message']:
            # Handle different message formats
            if ':' in message:
                # Split on first colon followed by space
                parts = message.split(': ', 1)
                if len(parts) == 2:
                    users.append(parts[0].strip())
                    messages.append(parts[1])
                else:
                    users.append('group_notification')
                    messages.append(message)
            else:
                users.append('group_notification')
                messages.append(message)

        df['user'] = users
        df['message'] = messages
        df.drop(columns=['user_message'], inplace=True)

        # Extract comprehensive time components
        df['only_date'] = df['date'].dt.date
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['month'] = df['date'].dt.month_name()
        df['day'] = df['date'].dt.day
        df['day_name'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Create time periods
        df['period'] = df['hour'].apply(lambda x: f"{x:02d}-{(x + 1) % 24:02d}")

        # Create time of day categories
        def get_time_period(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'

        df['time_of_day'] = df['hour'].apply(get_time_period)

        # Add message length and word count
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()

        # Identify message types
        df['is_media'] = df['message'].str.contains('<Media omitted>|<media omitted>', case=False, na=False)
        df['is_deleted'] = df['message'].str.contains('This message was deleted|deleted this message', case=False,
                                                      na=False)
        df['has_link'] = df['message'].apply(lambda x: len(extractor.find_urls(str(x))) > 0)
        df['has_emoji'] = df['message'].apply(lambda x: bool([c for c in str(x) if c in emoji.EMOJI_DATA]))

        return df

    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None


def get_comprehensive_stats(selected_user, df):
    """Get comprehensive statistics"""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    stats = {}

    # Basic stats
    stats['total_messages'] = len(df)
    stats['total_words'] = df['word_count'].sum()
    stats['media_messages'] = df['is_media'].sum()
    stats['deleted_messages'] = df['is_deleted'].sum()
    stats['messages_with_links'] = df['has_link'].sum()
    stats['messages_with_emojis'] = df['has_emoji'].sum()

    # Time-based stats
    stats['first_message'] = df['date'].min()
    stats['last_message'] = df['date'].max()
    stats['chat_duration'] = (stats['last_message'] - stats['first_message']).days
    stats['avg_messages_per_day'] = stats['total_messages'] / max(stats['chat_duration'], 1)

    # Message length stats
    stats['avg_message_length'] = df['message_length'].mean()
    stats['longest_message'] = df['message_length'].max()
    stats['avg_words_per_message'] = df['word_count'].mean()

    # Activity stats
    stats['most_active_day'] = df['day_name'].mode().iloc[0] if not df['day_name'].mode().empty else 'N/A'
    stats['most_active_hour'] = df['hour'].mode().iloc[0] if not df['hour'].mode().empty else 'N/A'
    stats['most_active_month'] = df['month'].mode().iloc[0] if not df['month'].mode().empty else 'N/A'

    return stats


def create_enhanced_wordcloud(selected_user, df):
    """Create enhanced word cloud with better filtering"""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Enhanced stop words
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'ours',
        'ke', 'che', 'ma', 'na', 'to', 'j', 'ko', 'kya', 'hai', 'hain', 'tha', 'thi', 'ka',
        'ki', 'se', 'me', 'par', 'pe', 'omitted', 'media', 'message', 'deleted'
    ])

    # Filter messages
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['is_media']]
    temp = temp[~temp['is_deleted']]
    temp = temp.dropna(subset=['message'])

    if temp.empty:
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
        return wc.generate("No text messages found")

    # Clean and prepare text
    text = ' '.join(temp['message'].astype(str))

    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s\u0900-\u097F\u0A80-\u0AFF]', '', text)

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10,
        stopwords=stop_words
    )

    return wc.generate(text)


def get_sentiment_analysis(selected_user, df):
    """Basic sentiment analysis using predefined word lists"""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Simple sentiment word lists
    positive_words = ['good', 'great', 'awesome', 'love', 'happy', 'excellent', 'amazing', 'best', 'wonderful',
                      'fantastic', 'yes', 'ok', 'thanks', 'thank', 'nice', 'cool', 'super', 'perfect', 'right', '👍',
                      '😊', '😍', '❤️', '🎉', '✨', '😁', '😄', '🥳']
    negative_words = ['bad', 'worst', 'hate', 'sad', 'terrible', 'awful', 'horrible', 'no', 'not', 'never', 'problem',
                      'issue', 'wrong', 'fail', 'failed', 'disappointed', '😢', '😞', '😠', '😡', '💔', '😭', '😪']

    temp = df[~df['is_media'] & ~df['is_deleted']].copy()
    temp['message_lower'] = temp['message'].str.lower()

    sentiment_scores = []
    for message in temp['message_lower']:
        if pd.isna(message):
            sentiment_scores.append(0)
            continue

        positive_count = sum(1 for word in positive_words if word in str(message))
        negative_count = sum(1 for word in negative_words if word in str(message))

        if positive_count > negative_count:
            sentiment_scores.append(1)  # Positive
        elif negative_count > positive_count:
            sentiment_scores.append(-1)  # Negative
        else:
            sentiment_scores.append(0)  # Neutral

    temp['sentiment'] = sentiment_scores

    sentiment_counts = temp['sentiment'].value_counts()
    sentiment_labels = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}

    return {sentiment_labels[k]: v for k, v in sentiment_counts.items()}


def create_interactive_timeline(selected_user, df):
    """Create interactive timeline using Plotly"""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Daily message count
    daily_data = df.groupby('only_date').size().reset_index(name='message_count')
    daily_data['only_date'] = pd.to_datetime(daily_data['only_date'])

    fig = px.line(daily_data, x='only_date', y='message_count',
                  title='Daily Message Activity',
                  labels={'only_date': 'Date', 'message_count': 'Messages'})
    fig.update_layout(hovermode='x unified')

    return fig


def create_activity_heatmap(selected_user, df):
    """Create activity heatmap"""
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(
        values='message',
        index='day_name',
        columns='hour',
        aggfunc='count',
        fill_value=0
    )

    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    fig = px.imshow(heatmap_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                    title="Weekly Activity Heatmap",
                    color_continuous_scale="Viridis")

    return fig


def get_emoji_analysis(selected_user, df):
    """Enhanced emoji analysis"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        if pd.notna(message):
            emojis.extend([c for c in str(message) if c in emoji.EMOJI_DATA])

    if not emojis:
        return pd.DataFrame(columns=['emoji', 'count'])

    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counts.most_common(20), columns=['emoji', 'count'])

    return emoji_df


def create_user_comparison(df):
    """Create user comparison charts"""
    user_stats = df[df['user'] != 'group_notification'].groupby('user').agg({
        'message': 'count',
        'word_count': 'sum',
        'is_media': 'sum',
        'has_link': 'sum',
        'has_emoji': 'sum',
        'message_length': 'mean'
    }).round(2)

    user_stats.columns = ['Messages', 'Words', 'Media', 'Links', 'Emojis', 'Avg Length']

    return user_stats


def get_conversation_insights(df):
    """Generate conversation insights"""
    insights = []

    # Most active user
    most_active = df[df['user'] != 'group_notification']['user'].value_counts().head(1)
    if not most_active.empty:
        insights.append(f"🏆 Most active user: **{most_active.index[0]}** with {most_active.values[0]} messages")

    # Peak activity time
    peak_hour = df['hour'].value_counts().head(1)
    if not peak_hour.empty:
        insights.append(f"⏰ Peak activity hour: **{peak_hour.index[0]}:00** with {peak_hour.values[0]} messages")

    # Busiest day
    busiest_day = df['day_name'].value_counts().head(1)
    if not busiest_day.empty:
        insights.append(f"📅 Busiest day: **{busiest_day.index[0]}** with {busiest_day.values[0]} messages")

    # Media sharing
    media_percentage = (df['is_media'].sum() / len(df)) * 100
    insights.append(f"📸 Media sharing: **{media_percentage:.1f}%** of all messages")

    # Average response time (simplified)
    avg_daily_messages = len(df) / max((df['date'].max() - df['date'].min()).days, 1)
    insights.append(f"💬 Average messages per day: **{avg_daily_messages:.1f}**")

    return insights


# Streamlit App
def main():
    st.markdown('<h1 class="main-header">📱 WhatsApp Chat Analyzer Pro</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Navigation")

    # File upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload Chat File (.txt)", type=['txt'])

    if uploaded_file is not None:
        try:
            # Show loading spinner
            with st.spinner('Processing chat data...'):
                data = uploaded_file.read().decode("utf-8")
                df = preprocess(data)

            if df is None:
                return

            st.success(f"✅ Chat loaded successfully! Found {len(df)} messages.")

            # User selection
            user_list = df['user'].unique().tolist()
            user_list = [user for user in user_list if user != "group_notification"]
            user_list.sort()
            user_list.insert(0, "Overall")

            selected_user = st.sidebar.selectbox("👤 Select User", user_list)

            # Analysis sections
            analysis_type = st.sidebar.selectbox(
                "📊 Analysis Type",
                ["📈 Overview", "💬 Message Analysis", "🕒 Time Analysis", "😀 Emoji & Sentiment", "👥 User Comparison",
                 "🔍 Insights"]
            )

            # Main content based on selection
            if analysis_type == "📈 Overview":
                show_overview(selected_user, df)
            elif analysis_type == "💬 Message Analysis":
                show_message_analysis(selected_user, df)
            elif analysis_type == "🕒 Time Analysis":
                show_time_analysis(selected_user, df)
            elif analysis_type == "😀 Emoji & Sentiment":
                show_emoji_sentiment_analysis(selected_user, df)
            elif analysis_type == "👥 User Comparison":
                show_user_comparison(df)
            elif analysis_type == "🔍 Insights":
                show_insights(df)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure you've uploaded a valid WhatsApp chat export file.")
    else:
        st.info("👆 Please upload a WhatsApp chat export file to begin analysis.")

        # Instructions
        with st.expander("📖 How to export WhatsApp chat"):
            st.markdown("""
            1. Open WhatsApp on your phone
            2. Go to the chat you want to analyze
            3. Tap on the contact/group name at the top
            4. Select 'Export Chat'
            5. Choose 'Without Media' for faster processing
            6. Share the .txt file to yourself and upload it here
            """)


def show_overview(selected_user, df):
    """Show overview statistics"""
    st.header(f"📊 Overview Analysis - {selected_user}")

    stats = get_comprehensive_stats(selected_user, df)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Total Messages", f"{stats['total_messages']:,}")
    with col2:
        st.metric("📝 Total Words", f"{stats['total_words']:,}")
    with col3:
        st.metric("📸 Media Messages", f"{stats['media_messages']:,}")
    with col4:
        st.metric("🔗 Links Shared", f"{stats['messages_with_links']:,}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("📅 Chat Duration", f"{stats['chat_duration']} days")
    with col6:
        st.metric("📊 Avg Messages/Day", f"{stats['avg_messages_per_day']:.1f}")
    with col7:
        st.metric("📏 Avg Message Length", f"{stats['avg_message_length']:.1f} chars")
    with col8:
        st.metric("🗣️ Avg Words/Message", f"{stats['avg_words_per_message']:.1f}")

    # Activity distribution
    if selected_user == "Overall":
        st.subheader("👥 User Activity Distribution")
        user_activity = df[df['user'] != 'group_notification']['user'].value_counts().head(10)

        fig = px.bar(x=user_activity.values, y=user_activity.index, orientation='h',
                     title="Top 10 Most Active Users",
                     labels={'x': 'Number of Messages', 'y': 'User'})
        st.plotly_chart(fig, use_container_width=True)


def show_message_analysis(selected_user, df):
    """Show message analysis"""
    st.header(f"💬 Message Analysis - {selected_user}")

    # Word cloud
    st.subheader("☁️ Word Cloud")
    wordcloud = create_enhanced_wordcloud(selected_user, df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Message types distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Message Types")
        if selected_user != "Overall":
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df

        message_types = {
            'Text': len(user_df[~user_df['is_media'] & ~user_df['is_deleted']]),
            'Media': user_df['is_media'].sum(),
            'Deleted': user_df['is_deleted'].sum(),
            'With Links': user_df['has_link'].sum(),
            'With Emojis': user_df['has_emoji'].sum()
        }

        fig = px.pie(values=list(message_types.values()), names=list(message_types.keys()),
                     title="Message Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Message Length Distribution")
        lengths = user_df[user_df['message_length'] > 0]['message_length']

        fig = px.histogram(x=lengths, bins=30, title="Message Length Distribution",
                           labels={'x': 'Message Length (characters)', 'y': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)


def show_time_analysis(selected_user, df):
    """Show time-based analysis"""
    st.header(f"🕒 Time Analysis - {selected_user}")

    # Interactive timeline
    st.subheader("📈 Daily Activity Timeline")
    timeline_fig = create_interactive_timeline(selected_user, df)
    st.plotly_chart(timeline_fig, use_container_width=True)

    # Activity heatmap
    st.subheader("🔥 Weekly Activity Heatmap")
    heatmap_fig = create_activity_heatmap(selected_user, df)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Time of day analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌅 Activity by Time of Day")
        if selected_user != "Overall":
            time_data = df[df['user'] == selected_user]['time_of_day'].value_counts()
        else:
            time_data = df['time_of_day'].value_counts()

        fig = px.bar(x=time_data.index, y=time_data.values,
                     title="Messages by Time of Day",
                     labels={'x': 'Time Period', 'y': 'Number of Messages'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Monthly Activity")
        if selected_user != "Overall":
            monthly_data = df[df['user'] == selected_user]['month'].value_counts()
        else:
            monthly_data = df['month'].value_counts()

        fig = px.bar(x=monthly_data.index, y=monthly_data.values,
                     title="Messages by Month",
                     labels={'x': 'Month', 'y': 'Number of Messages'})
        st.plotly_chart(fig, use_container_width=True)


def show_emoji_sentiment_analysis(selected_user, df):
    """Show emoji and sentiment analysis"""
    st.header(f"😀 Emoji & Sentiment Analysis - {selected_user}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("😊 Top Emojis")
        emoji_df = get_emoji_analysis(selected_user, df)

        if not emoji_df.empty:
            st.dataframe(emoji_df.head(10), use_container_width=True)

            # Emoji pie chart
            fig = px.pie(emoji_df.head(10), values='count', names='emoji',
                         title="Most Used Emojis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emojis found in the selected data.")

    with col2:
        st.subheader("💭 Sentiment Analysis")
        sentiment_data = get_sentiment_analysis(selected_user, df)

        if sentiment_data:
            fig = px.pie(values=list(sentiment_data.values()), names=list(sentiment_data.keys()),
                         title="Message Sentiment Distribution",
                         color_discrete_map={'Positive': '#00CC96', 'Neutral': '#AB63FA', 'Negative': '#FF6692'})
            st.plotly_chart(fig, use_container_width=True)

            # Sentiment stats
            total_messages = sum(sentiment_data.values())
            for sentiment, count in sentiment_data.items():
                percentage = (count / total_messages) * 100
                st.metric(f"{sentiment} Messages", f"{count} ({percentage:.1f}%)")


def show_user_comparison(df):
    """Show user comparison analysis"""
    st.header("👥 User Comparison Analysis")

    user_stats = create_user_comparison(df)

    if len(user_stats) > 1:
        st.subheader("📊 User Statistics Comparison")
        st.dataframe(user_stats, use_container_width=True)

        # Create comparison charts
        metrics = ['Messages', 'Words', 'Media', 'Links', 'Emojis']

        for i, metric in enumerate(metrics):
            if i % 2 == 0:
                col1, col2 = st.columns(2)
                current_col = col1
            else:
                current_col = col2

            with current_col:
                fig = px.bar(x=user_stats.index, y=user_stats[metric],
                             title=f"{metric} by User",
                             labels={'x': 'User', 'y': metric})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 users for comparison analysis.")


def show_insights(df):
    """Show conversation insights"""
    st.header("🔍 Conversation Insights")

    insights = get_conversation_insights(df)

    st.subheader("💡 Key Insights")
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # Additional statistics
    st.subheader("📋 Detailed Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Users", len(df['user'].unique()) - 1)  # Exclude group_notification
        st.metric("First Message", df['date'].min().strftime('%Y-%m-%d'))
        st.metric("Latest Message", df['date'].max().strftime('%Y-%m-%d'))

    with col2:
        st.metric("Total Days Active", (df['date'].max() - df['date'].min()).days)
        st.metric("Messages with Emojis", df['has_emoji'].sum())
        st.metric("Deleted Messages", df['is_deleted'].sum())

    with col3:
        avg_response_time = len(df) / max((df['date'].max() - df['date'].min()).days, 1)
        st.metric("Avg Messages/Day", f"{avg_response_time:.1f}")
        st.metric("Peak Activity Hour", df['hour'].mode().iloc[0] if not df['hour'].mode().empty else 'N/A')
        st.metric("Most Active Day", df['day_name'].mode().iloc[0] if not df['day_name'].mode().empty else 'N/A')

    # Conversation flow analysis
    st.subheader("🔄 Conversation Flow")

    # Messages per hour heatmap
    hourly_activity = df.groupby(['day_name', 'hour']).size().reset_index(name='message_count')

    if not hourly_activity.empty:
        pivot_hourly = hourly_activity.pivot(index='day_name', columns='hour', values='message_count').fillna(0)

        fig = px.imshow(pivot_hourly,
                        labels=dict(x="Hour", y="Day", color="Messages"),
                        title="Hourly Activity Pattern",
                        color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    # Response patterns (simplified)
    st.subheader("⚡ Quick Stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        longest_msg = df.loc[df['message_length'].idxmax()] if not df.empty else None
        if longest_msg is not None:
            st.metric("Longest Message", f"{longest_msg['message_length']} chars")
            with st.expander("Show longest message"):
                st.text(longest_msg['message'][:200] + "..." if len(longest_msg['message']) > 200 else longest_msg[
                    'message'])

    with col2:
        media_percentage = (df['is_media'].sum() / len(df)) * 100
        st.metric("Media Share %", f"{media_percentage:.1f}%")

    with col3:
        link_percentage = (df['has_link'].sum() / len(df)) * 100
        st.metric("Links Share %", f"{link_percentage:.1f}%")

    with col4:
        emoji_percentage = (df['has_emoji'].sum() / len(df)) * 100
        st.metric("Emoji Usage %", f"{emoji_percentage:.1f}%")


def create_advanced_analytics(selected_user, df):
    """Create advanced analytics dashboard"""
    st.header("🚀 Advanced Analytics")

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Message frequency analysis
    st.subheader("📊 Message Frequency Analysis")

    # Daily message distribution
    daily_msgs = df.groupby('only_date').size()

    col1, col2 = st.columns(2)

    with col1:
        # Message frequency histogram
        fig = px.histogram(x=daily_msgs.values, nbins=20,
                           title="Daily Message Count Distribution",
                           labels={'x': 'Messages per Day', 'y': 'Number of Days'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cumulative messages over time
        df_sorted = df.sort_values('date')
        df_sorted['cumulative_messages'] = range(1, len(df_sorted) + 1)

        fig = px.line(df_sorted, x='date', y='cumulative_messages',
                      title="Cumulative Messages Over Time",
                      labels={'date': 'Date', 'cumulative_messages': 'Total Messages'})
        st.plotly_chart(fig, use_container_width=True)

    # Weekly patterns
    st.subheader("📅 Weekly Patterns")

    # Create week-over-week comparison
    df['week_year'] = df['date'].dt.isocalendar().week.astype(str) + '-' + df['date'].dt.year.astype(str)
    weekly_data = df.groupby(['week_year', 'day_name']).size().reset_index(name='message_count')

    if len(weekly_data) > 0:
        # Average messages by day of week
        avg_by_day = df.groupby('day_name').size().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).fillna(0)

        fig = px.bar(x=avg_by_day.index, y=avg_by_day.values,
                     title="Average Messages by Day of Week",
                     labels={'x': 'Day', 'y': 'Average Messages'})
        st.plotly_chart(fig, use_container_width=True)


def create_export_report(selected_user, df):
    """Create exportable report"""
    st.header("📄 Export Report")

    stats = get_comprehensive_stats(selected_user, df)

    # Create report content
    report_content = f"""
# WhatsApp Chat Analysis Report
**Analysis for:** {selected_user}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Messages:** {stats['total_messages']:,}
- **Total Words:** {stats['total_words']:,}
- **Media Messages:** {stats['media_messages']:,}
- **Messages with Links:** {stats['messages_with_links']:,}
- **Messages with Emojis:** {stats['messages_with_emojis']:,}
- **Chat Duration:** {stats['chat_duration']} days
- **Average Messages per Day:** {stats['avg_messages_per_day']:.2f}
- **Average Message Length:** {stats['avg_message_length']:.2f} characters
- **Most Active Day:** {stats['most_active_day']}
- **Most Active Hour:** {stats['most_active_hour']}:00
- **Most Active Month:** {stats['most_active_month']}

## Activity Insights
- **First Message:** {stats['first_message'].strftime('%Y-%m-%d %H:%M')}
- **Last Message:** {stats['last_message'].strftime('%Y-%m-%d %H:%M')}
- **Longest Message:** {stats['longest_message']} characters
- **Average Words per Message:** {stats['avg_words_per_message']:.2f}

## Top Emojis Used
"""

    # Add emoji data to report
    emoji_data = get_emoji_analysis(selected_user, df)
    if not emoji_data.empty:
        for idx, row in emoji_data.head(10).iterrows():
            report_content += f"- {row['emoji']}: {row['count']} times\n"
    else:
        report_content += "- No emojis found\n"

    # Add sentiment analysis
    sentiment_data = get_sentiment_analysis(selected_user, df)
    if sentiment_data:
        report_content += "\n## Sentiment Analysis\n"
        total_analyzed = sum(sentiment_data.values())
        for sentiment, count in sentiment_data.items():
            percentage = (count / total_analyzed) * 100
            report_content += f"- **{sentiment}:** {count} messages ({percentage:.1f}%)\n"

    report_content += f"\n---\n*Report generated by WhatsApp Chat Analyzer Pro*"

    # Display report
    st.markdown(report_content)

    # Download button
    st.download_button(
        label="📥 Download Report",
        data=report_content,
        file_name=f"whatsapp_analysis_{selected_user}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )


# Enhanced main function with all features
def main():
    st.markdown('<h1 class="main-header">📱 WhatsApp Chat Analyzer Pro</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📋 Navigation")

    # File upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload Chat File (.txt)", type=['txt'])

    if uploaded_file is not None:
        try:
            # Show loading spinner
            with st.spinner('Processing chat data...'):
                data = uploaded_file.read().decode("utf-8")
                df = preprocess(data)

            if df is None:
                return

            st.success(
                f"✅ Chat loaded successfully! Found {len(df)} messages from {len(df['user'].unique()) - 1} users.")

            # User selection
            user_list = df['user'].unique().tolist()
            user_list = [user for user in user_list if user != "group_notification"]
            user_list.sort()
            user_list.insert(0, "Overall")

            selected_user = st.sidebar.selectbox("👤 Select User", user_list)

            # Analysis sections
            analysis_type = st.sidebar.selectbox(
                "📊 Analysis Type",
                [
                    "📈 Overview",
                    "💬 Message Analysis",
                    "🕒 Time Analysis",
                    "😀 Emoji & Sentiment",
                    "👥 User Comparison",
                    "🔍 Insights",
                    "🚀 Advanced Analytics",
                    "📄 Export Report"
                ]
            )

            # Main content based on selection
            if analysis_type == "📈 Overview":
                show_overview(selected_user, df)
            elif analysis_type == "💬 Message Analysis":
                show_message_analysis(selected_user, df)
            elif analysis_type == "🕒 Time Analysis":
                show_time_analysis(selected_user, df)
            elif analysis_type == "😀 Emoji & Sentiment":
                show_emoji_sentiment_analysis(selected_user, df)
            elif analysis_type == "👥 User Comparison":
                show_user_comparison(df)
            elif analysis_type == "🔍 Insights":
                show_insights(df)
            elif analysis_type == "🚀 Advanced Analytics":
                create_advanced_analytics(selected_user, df)
            elif analysis_type == "📄 Export Report":
                create_export_report(selected_user, df)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure you've uploaded a valid WhatsApp chat export file.")

            # Debug information
            with st.expander("🔧 Debug Information"):
                st.code(str(e))
                st.write("If the error persists, please check:")
                st.write("1. File format is correct (.txt)")
                st.write("2. Chat export includes timestamps")
                st.write("3. File is not corrupted")

    else:
        # Welcome screen
        st.info("👆 Please upload a WhatsApp chat export file to begin analysis.")

        # Feature overview
        st.markdown("""
        ## 🌟 Features

        **📈 Overview Analysis**
        - Comprehensive chat statistics
        - User activity metrics
        - Timeline analysis

        **💬 Message Analysis**
        - Word cloud generation
        - Message type distribution
        - Content analysis

        **🕒 Time Analysis**
        - Interactive daily timeline
        - Weekly activity heatmap
        - Peak activity identification

        **😀 Emoji & Sentiment**
        - Most used emojis
        - Sentiment analysis
        - Emotional patterns

        **👥 User Comparison**
        - Multi-user statistics
        - Activity comparison
        - Engagement metrics

        **🔍 Smart Insights**
        - AI-powered insights
        - Conversation patterns
        - Behavioral analysis

        **🚀 Advanced Analytics**
        - Message frequency analysis
        - Weekly patterns
        - Cumulative trends

        **📄 Export Reports**
        - Downloadable analysis
        - Comprehensive summaries
        - Markdown format
        """)

        # Instructions
        with st.expander("📖 How to export WhatsApp chat"):
            st.markdown("""
            ### For Android:
            1. Open WhatsApp and go to the chat
            2. Tap the three dots menu → More → Export chat
            3. Choose 'Without Media' for faster processing
            4. Share the .txt file to yourself

            ### For iPhone:
            1. Open WhatsApp and go to the chat
            2. Tap the contact/group name at the top
            3. Scroll down and tap 'Export Chat'
            4. Choose 'Without Media'
            5. Share the .txt file to yourself

            ### Tips:
            - File size should be under 200MB for best performance
            - Larger chats may take longer to process
            - Export without media for faster analysis
            """)


if __name__ == "__main__":
    main()