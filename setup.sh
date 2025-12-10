mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor = '#E50914'\n\
backgroundColor = '#141414'\n\
secondaryBackgroundColor = '#1a1a2e'\n\
textColor = '#ffffff'\n\
" > ~/.streamlit/config.toml
