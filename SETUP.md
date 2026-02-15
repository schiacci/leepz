# Quick Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Get an OpenRouter API Key

1. Go to https://openrouter.ai
2. Sign up for an account
3. Generate an API key
4. Copy your API key

## 3. Configure Environment

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:
```bash
OPENROUTER_API_KEY=your_key_here
```

**Auto-activation (Optional)**: The project includes `.envrc` for automatic virtual environment activation when entering the directory:
```bash
# Install direnv (optional)
brew install direnv  # macOS
direnv allow         # Allow auto-activation
```

## 4. Create Directories

```bash
mkdir -p database logs outputs
```

## 5. Test the Installation

```bash
# Test with a single ticker
python main.py --ticker AAPL --narrative "Test run"
```

## 6. Verify Security Setup

Before committing to GitHub, verify your sensitive data is protected:

```bash
git status --ignored
```

You should see `.env`, `.envrc`, `database/`, `outputs/`, and `leap_env/` in the "Ignored files" section.

## 7. Run Full Pipeline

```bash
python main.py
```

## Troubleshooting

### "OPENROUTER_API_KEY not found"
- Make sure you created `.env` file
- Make sure the API key is on the line `OPENROUTER_API_KEY=sk-or-...`
- No quotes needed around the key

### "No module named 'yfinance'"
- Run `pip install -r requirements.txt`

### "No suitable LEAP options found"
- Some stocks don't have LEAP options (need to be liquid, large-cap)
- Try major tickers like AAPL, MSFT, NVDA, GOOGL

### yfinance errors
- yfinance occasionally has rate limits
- Wait a few seconds between tickers
- Some tickers may not have complete option chain data

### Git Status Shows Sensitive Files
- Check that `.gitignore` exists and contains the correct patterns
- Run `git rm --cached .env` if accidentally added
- Ensure you're not in the virtual environment when checking

### Virtual Environment Auto-activation Not Working
- Install `direnv`: `brew install direnv` (macOS)
- Run `direnv allow` in the project directory
- Restart your terminal

## Next Steps

1. Read the full README.md
2. Check out examples.py for programmatic usage
3. Explore the database schema in database/db.py
4. Customize LEAP parameters in .env

## Support

- Check the README.md for detailed documentation
- Review the code comments
- Test with well-known, liquid tickers first
