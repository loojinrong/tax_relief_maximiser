# Singapore Tax Relief Calculator

A web application that helps Singapore residents calculate their tax reliefs and optimize their tax savings using AI-powered recommendations.

## Features

- **Interactive Tax Calculator**: Calculate your tax liability based on various income sources and reliefs
- **AI-Powered Recommendations**: Get personalized tax-saving recommendations using AI
- **Comprehensive Relief Coverage**: Supports multiple tax relief schemes including:
  - Earned Income Relief
  - Parent Relief
  - Child Relief
  - Working Mother's Child Relief
  - CPF Relief
  - SRS Relief
  - Course Fee Relief
  - NSman Relief
  - Donation Relief
  - Parenthood Tax Rebate
- **Interactive Chatbot**: Get answers to your tax-related questions
- **User-Friendly Interface**: Step-by-step form to input your financial information

## Prerequisites

- Python 3.7 or higher
- OpenAI API key (for AI recommendations and chatbot)
- Conda or pip for package management

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tax_relief_again
```

2. Create and activate a virtual environment:
```bash
conda create --name tax_env python=3.7
conda activate tax_env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Follow the step-by-step form to input your financial information:
   - General Information
   - Income Details
   - Relief Information
   - Children Information (if applicable)

4. View your tax calculation results and AI recommendations

## Project Structure

```
tax_relief_again/
├── app.py              # Main Flask application
├── forms.py            # Form definitions using WTForms
├── models.py           # Tax calculation models
├── tax_calculator.py   # Tax calculation utilities
├── requirements.txt    # Project dependencies
├── .env                # Environment variables
├── templates/          # HTML templates
├── static/             # Static files (CSS, JS, images)
└── README.md           # Project documentation
```

## Dependencies

- Flask: Web framework
- OpenAI: AI-powered recommendations and chatbot
- python-dotenv: Environment variable management
- torch: Machine learning framework
- transformers: Natural language processing
- Flask-WTF: Form handling
- WTForms: Form validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Singapore Inland Revenue Authority (IRAS) for tax relief information
- OpenAI for AI capabilities
- Flask community for the web framework 