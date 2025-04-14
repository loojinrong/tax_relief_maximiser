##############################################
# File: app.py (Updated with AI integration)
##############################################

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import math, os, json
import openai
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from forms import GeneralInfoForm, IncomeForm, ReliefForm, ChildrenInfoForm
from models import (
    calculate_tax, calculate_earned_income_relief, calculate_parent_relief,
    calculate_child_relief, calculate_working_mother_child_relief,
    calculate_cpf_relief, calculate_srs_relief, calculate_donation_relief,
    calculate_course_fee_relief, calculate_nsman_relief, calculate_parenthood_tax_rebate
)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = '457314cbd7778d17f1795670862ab03f160afceb01772b43fe658f1ba0d4c6c5'  # Secure secret key for session encryption

# Set your OpenAI API key (ensure it's set in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model and tokenizer
tokenizer, model = None, None

RELIEF_CAP = 80000

BRACKETS = [
    20000,   # Bracket 0: up to 20k
    30000,   # Bracket 1: 20k -> 30k
    40000,   # Bracket 2: 30k -> 40k
    80000,   # Bracket 3: 40k -> 80k
    120000,  # Bracket 4: 80k -> 120k
    160000,  # Bracket 5: 120k -> 160k
    200000,  # Bracket 6: 160k -> 200k
    240000,  # Bracket 7: 200k -> 240k
    280000,  # Bracket 8: 240k -> 280k
    320000,  # Bracket 9: 280k -> 320k
    500000,  # Bracket 10: 320k -> 500k
    1000000, # Bracket 11: 500k -> 1M
    float('inf')  # Bracket 12: Above 1M
]

def load_model(model_name="EleutherAI/gpt-neo-2.7B"):
    """
    Loads the tokenizer and model from Hugging Face.
    You can change the model_name to a larger model (e.g., "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-2.7B")
    if you have sufficient resources.
    """
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_recommendations(user_data, tokenizer, model, max_length=300):
    """
    Generates tax-saving recommendations using the loaded model.
    """
    prompt = f"""
You are a tax savings expert specializing in Singapore taxation. Based on the following user data, generate a JSON array of tax-saving recommendations. Each recommendation must include:
- "Relief Name": a short title
- "Suggested Amount": a numerical value representing the additional relief amount
- "Rationale": an explanation of why the strategy is beneficial

User Data:
- Employment Income: {user_data['employment_income']}
- Employment Expenses: {user_data['employment_expenses']}
- Trade, Business, Profession, and Vocation Income: {user_data['trade_business_income']}
- Dividends: {user_data['dividends']}
- Interest: {user_data['interest']}
- Net Rent from Property: {user_data['rent']}
- Approved Donations: {user_data['approved_donations']}
- Earned Income Relief (EIR): {user_data['eir']}
- Parent Relief: {user_data['parent_relief']}
- CPF Relief: {user_data['cpf_relief']}
- SRS Relief: {user_data['srs_relief']}
- Course Fee Relief: {user_data['course_fee_relief']}
- Current Chargeable Income: {user_data['chargeable_income']}

Please output only the JSON array.
"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # If a GPU is available, move the tensors and model to CUDA
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.to("cuda")
    
    # Generate output using the model
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Sometimes the output contains the original prompt, so extract only the JSON array.
    start_idx = output_text.find('[')
    if start_idx == -1:
        print("JSON output not found in generated text.")
        return []
    
    json_text = output_text[start_idx:]
    try:
        recommendations = json.loads(json_text)
    except Exception as e:
        print("Error parsing JSON from model output:", e)
        recommendations = []
    return recommendations

def ai_generate_recommendations(user_data):
    """
    Given the user data, generate personalized tax-saving recommendations.
    This function will use the local model if available, otherwise fall back to OpenAI.
    """
    global tokenizer, model
    
    # Initialize the model if not already loaded
    if tokenizer is None or model is None:
        try:
            tokenizer, model = load_model()
        except Exception as e:
            print("Error loading local model:", e)
            # Fall back to OpenAI if local model fails
            return ai_generate_recommendations_openai(user_data)
    
    try:
        return generate_recommendations(user_data, tokenizer, model)
    except Exception as e:
        print("Error with local model:", e)
        # Fall back to OpenAI if local model fails
        return ai_generate_recommendations_openai(user_data)

def ai_generate_recommendations_openai(user_data):
    """
    Generate tax-saving recommendations with detailed savings calculations
    """
    current_tax = calculate_tax(user_data['chargeable_income'])
    
    prompt = f"""
You are a tax savings expert specializing in Singapore taxation. Your task is to generate SPECIFIC tax-saving recommendations with exact calculations.

CURRENT STATUS:
- Chargeable Income: ${user_data['chargeable_income']}
- Current Tax Payable: ${current_tax}
- Available Relief Space: ${80000 - user_data.get('total_reliefs', 0)} (up to $80,000 total relief cap)

CURRENT RELIEFS USED:
- CPF Relief: ${user_data['self_top_up_cpf']} (max: $8,000)
- SRS Relief: ${user_data['srs_relief']} (max: $15,300)
- Course Fee Relief: ${user_data['course_fee_relief']} (max: $5,500)
- Donations: ${user_data['approved_donations']}

REQUIRED: Generate 3-4 specific recommendations. For each recommendation, you MUST calculate:
1. Exact amount to contribute
2. Tax deduction value (e.g., 2.5x for donations)
3. New chargeable income after deduction
4. New tax payable
5. Total tax savings

YOU MUST OUTPUT IN THIS EXACT JSON FORMAT:
[
    {{
        "Relief Name": "CPF Top-up",
        "Suggested Amount": 5000,
        "Tax Deduction": 5000,
        "Current Tax": 3000,
        "New Tax": 2500,
        "Tax Savings": 500,
        "Rationale": "Making a $5,000 CPF top-up will reduce your tax by $500 while boosting retirement savings"
    }},
    {{
        "Relief Name": "Charitable Donation",
        "Suggested Amount": 2000,
        "Tax Deduction": 5000,
        "Current Tax": 3000,
        "New Tax": 2200,
        "Tax Savings": 800,
        "Rationale": "A $2,000 donation provides $5,000 tax deduction (2.5x) and saves $800 in tax"
    }}
]

IMPORTANT RULES:
1. All amounts must be specific numbers, not ranges or descriptions
2. Each suggestion must include exact tax savings calculation
3. Rationale must explain both tax savings and additional benefits
4. Consider remaining relief cap when making suggestions
5. Output must be valid JSON array with all fields specified above
6. DO NOT include any text outside the JSON array

STRATEGIES TO CONSIDER:
1. CPF Top-up (if below $8,000 cap)
2. SRS Contribution (if below $15,300 cap)
3. Approved Donations (2.5x tax deduction)
4. Course Fee Relief (if below $5,500 cap)

Generate recommendations now, output as JSON array only:"""

    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a tax expert. You MUST respond with ONLY a JSON array containing tax recommendations. 
                 Each recommendation MUST include exact numerical calculations and all required fields. 
                 DO NOT include any text outside the JSON array."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Extract JSON content
        content = response.choices[0].message.content.strip()
        
        # If the response contains any explanatory text, try to extract just the JSON part
        if content.startswith('[') and content.endswith(']'):
            recommendations = json.loads(content)
        else:
            # If no valid JSON array is found, create default recommendations
            recommendations = [
                {
                    "Relief Name": "CPF Top-up",
                    "Suggested Amount": min(8000 - user_data['self_top_up_cpf'], 8000),
                    "Tax Deduction": min(8000 - user_data['self_top_up_cpf'], 8000),
                    "Current Tax": current_tax,
                    "New Tax": calculate_tax(user_data['chargeable_income'] - min(8000 - user_data['self_top_up_cpf'], 8000)),
                    "Tax Savings": current_tax - calculate_tax(user_data['chargeable_income'] - min(8000 - user_data['self_top_up_cpf'], 8000)),
                    "Rationale": f"Maximize your CPF relief by contributing ${min(8000 - user_data['self_top_up_cpf'], 8000):,.2f}. This reduces your tax while securing your retirement."
                },
                {
                    "Relief Name": "SRS Contribution",
                    "Suggested Amount": min(15300 - user_data['srs_relief'], 15300),
                    "Tax Deduction": min(15300 - user_data['srs_relief'], 15300),
                    "Current Tax": current_tax,
                    "New Tax": calculate_tax(user_data['chargeable_income'] - min(15300 - user_data['srs_relief'], 15300)),
                    "Tax Savings": current_tax - calculate_tax(user_data['chargeable_income'] - min(15300 - user_data['srs_relief'], 15300)),
                    "Rationale": f"Open an SRS account and contribute ${min(15300 - user_data['srs_relief'], 15300):,.2f} to reduce your taxes while saving for retirement."
                },
                {
                    "Relief Name": "Charitable Donation",
                    "Suggested Amount": 2000,
                    "Tax Deduction": 5000,
                    "Current Tax": current_tax,
                    "New Tax": calculate_tax(user_data['chargeable_income'] - 5000),
                    "Tax Savings": current_tax - calculate_tax(user_data['chargeable_income'] - 5000),
                    "Rationale": "Donate $2,000 to receive a $5,000 tax deduction (2.5x) while supporting charitable causes."
                }
            ]
        
        # Validate and clean recommendations
        for rec in recommendations:
            for key in ["Suggested Amount", "Tax Deduction", "Current Tax", "New Tax", "Tax Savings"]:
                rec[key] = float(rec[key])  # Ensure numerical values are floats
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

def calculate_tax(income):
    # Using a simplified version of the bracket-based calculation.
    tax_brackets = [
        (1000000, 0.0, 199150), (1000000, 24.0, 0),
        (500000, 0.0, 84150), (10000, 23.0, 115000),
        (320000, 0.0, 44550), (180000, 22.0, 39600),
        (280000, 0.0, 36550), (40000, 20.0, 8000),
        (240000, 0.0, 28750), (40000, 19.5, 7800),
        (200000, 0.0, 21150), (40000, 19.0, 7600),
        (160000, 0.0, 13950), (40000, 18.0, 7200),
        (120000, 0.0, 7950), (40000, 15.0, 6000),
        (80000, 0.0, 3350), (40000, 11.5, 4600),
        (40000, 0.0, 550), (40000, 7.0, 2800),
        (30000, 0.0, 200), (10000, 3.5, 350),
        (20000, 0.0, 0), (10000, 2.0, 200)
    ]
    
    tax_payable = 0
    remaining_income = 0
    for i in range(0, len(tax_brackets), 2):
        if income > tax_brackets[i][0]:
            remaining_income = income - tax_brackets[i][0]
            tax_payable = tax_brackets[i][2] + (remaining_income * (tax_brackets[i+1][1] / 100.0))
            rebate = min(tax_payable * 0.6, 200)  # 60% rebate capped at $200
            tax_payable -= rebate
            break
    return max(0, tax_payable)

def get_bracket_index(income):
    """Return the index of the bracket in which the income falls."""
    for idx, threshold in enumerate(BRACKETS):
        if income <= threshold:
            return idx
    return len(BRACKETS) - 1

# ------------------------------
# Flask Routes (Summary)
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/general-questions', methods=['GET', 'POST'])
def general_questions():
    form = GeneralInfoForm()
    if form.validate_on_submit():
        session['citizenship'] = form.citizenship.data
        session['gender'] = form.gender.data
        session['marital_status'] = form.marital_status.data
        session['age_2025'] = form.age_2025.data
        session['disability'] = form.disability.data
        session['num_children'] = form.num_children.data
        session['num_handicapped_children'] = form.num_handicapped_children.data
        
        if session['num_children'] > 0:
            return redirect(url_for('children_info'))
        else:
            return redirect(url_for('income_questions'))
    
    return render_template('general_questions.html', form=form)

@app.route('/children-info', methods=['GET', 'POST'])
def children_info():
    num_children = session.get('num_children', 0)
    form = ChildrenInfoForm(num_children=num_children)
    
    if form.validate_on_submit():
        children_data = []
        for child_form in form.children:
            child_data = {
                'birth_year': child_form.birth_year.data,
                'birth_order': child_form.birth_order.data,
                'is_handicapped': child_form.is_handicapped.data
            }
            children_data.append(child_data)
        session['children_data'] = children_data
        return redirect(url_for('income_questions'))
    
    return render_template('children_info.html', form=form)

@app.route('/income-questions', methods=['GET', 'POST'])
def income_questions():
    form = IncomeForm()
    if form.validate_on_submit():
        session['employment_income'] = form.employment_income.data
        session['employment_expenses'] = form.employment_expenses.data
        session['trade_business_income'] = form.trade_business_income.data
        session['dividends'] = form.dividends.data
        session['interest'] = form.interest.data
        session['rent'] = form.rent.data
        session['approved_donations'] = form.approved_donations.data
        return redirect(url_for('relief_questions'))
    
    return render_template('income_questions.html', form=form)

@app.route('/relief-questions', methods=['GET', 'POST'])
def relief_questions():
    form = ReliefForm()
    if form.validate_on_submit():
        # Store relief information in session
        session['parent_relief_type'] = form.parent_relief_type.data
        session['num_parent_dependents'] = form.num_parent_dependents.data
        session['parent_living_with_you'] = form.parent_living_with_you.data
        session['cpf_contributions'] = form.cpf_contributions.data
        session['self_top_up_cpf'] = form.self_top_up_cpf.data
        session['srs_contributions'] = form.srs_contributions.data
        session['course_fees'] = form.course_fees.data
        session['nsman_type'] = form.nsman_type.data
        session['performed_ns'] = form.performed_ns.data
        session['is_key_appointment'] = form.is_key_appointment.data
        
        return redirect(url_for('results'))
    
    return render_template('relief_questions.html', form=form)

@app.route('/results')
def results():
    # Calculate total income
    income = max(0, session.get('employment_income', 0) - session.get('employment_expenses', 0))
    other_income = (session.get('dividends', 0) + 
                   session.get('interest', 0) + 
                   session.get('rent', 0))
    total_income = income + session.get('trade_business_income', 0) + other_income
    
    print(f"Debug - Income calculations:")
    print(f"Employment income: {session.get('employment_income', 0)}")
    print(f"Employment expenses: {session.get('employment_expenses', 0)}")
    print(f"Trade, Business, Profession, and Vocation Income: {session.get('trade_business_income', 0)}")
    print(f"Other income: {other_income}")
    print(f"Total income: {total_income}")
    
    # Calculate assessable income
    donation_relief = calculate_donation_relief(session.get('approved_donations', 0))
    assessable_income = max(0, total_income - donation_relief)
    
    print(f"\nDebug - Assessable income calculations:")
    print(f"Donation relief: {donation_relief}")
    print(f"Assessable income: {assessable_income}")
    
    # Calculate personal reliefs
    earned_income_relief = calculate_earned_income_relief(
        session.get('age_2025', 0),
        income
    )
    
    parent_relief = calculate_parent_relief(
        session.get('parent_relief_type', 'none'),
        session.get('num_parent_dependents', 0),
        session.get('parent_living_with_you', 'no') == 'yes'
    )
    
    child_relief = calculate_child_relief(
        session.get('num_children', 0),
        session.get('num_handicapped_children', 0)
    )
    
    working_mother_relief = calculate_working_mother_child_relief(
        session.get('num_children', 0)
    )
    
    cpf_relief = calculate_cpf_relief(session.get('cpf_contributions', 0))
    srs_relief = calculate_srs_relief(
        session.get('srs_contributions', 0),
        session.get('citizenship') in ['citizen', 'pr']
    )
    course_fee_relief = calculate_course_fee_relief(session.get('course_fees', 0))
    
    nsman_relief = calculate_nsman_relief(
        session.get('nsman_type', 'none'),
        session.get('performed_ns', 'no') == 'yes',
        session.get('is_key_appointment', 'no') == 'yes'
    )
    
    print(f"\nDebug - Relief calculations:")
    print(f"Earned income relief: {earned_income_relief}")
    print(f"Parent relief: {parent_relief}")
    print(f"Child relief: {child_relief}")
    print(f"Working mother relief: {working_mother_relief}")
    print(f"CPF relief: {cpf_relief}")
    print(f"SRS relief: {srs_relief}")
    print(f"Course fee relief: {course_fee_relief}")
    print(f"NSman relief: {nsman_relief}")
    
    # Calculate total reliefs
    total_reliefs = (
        earned_income_relief +
        parent_relief +
        child_relief +
        working_mother_relief +
        cpf_relief +
        srs_relief +
        course_fee_relief +
        nsman_relief
    )
    
    # Apply relief cap
    total_reliefs = min(total_reliefs, 80000)
    
    print(f"\nDebug - Total reliefs: {total_reliefs}")
    
    # Calculate chargeable income
    chargeable_income = max(0, assessable_income - total_reliefs)
    
    print(f"\nDebug - Chargeable income: {chargeable_income}")
    
    # Calculate tax
    tax_payable = calculate_tax(chargeable_income)
    
    print(f"\nDebug - Tax payable: {tax_payable}")
    
    # Calculate parenthood tax rebate
    parenthood_rebate = calculate_parenthood_tax_rebate(session.get('children_data', []))
    
    # Calculate final tax payable
    final_tax_payable = max(0, tax_payable - parenthood_rebate)
    
    print(f"\nDebug - Final tax payable: {final_tax_payable}")
    
    # Get AI recommendations
    user_data = {
        'employment_income': session.get('employment_income', 0),
        'employment_expenses': session.get('employment_expenses', 0),
        'trade_business_income': session.get('trade_business_income', 0),
        'dividends': session.get('dividends', 0),
        'interest': session.get('interest', 0),
        'rent': session.get('rent', 0),
        'approved_donations': session.get('approved_donations', 0),
        'eir': earned_income_relief,
        'parent_relief': parent_relief,
        'cpf_relief': cpf_relief,
        'srs_relief': srs_relief,
        'course_fee_relief': course_fee_relief,
        'chargeable_income': chargeable_income
    }
    
    recommended = ai_generate_recommendations_openai(user_data)
    
    # Calculate total potential savings
    total_potential_savings = sum(item['Tax Savings'] for item in recommended) if recommended else 0
    
    # Calculate new tax after applying all recommendations
    new_tax = min(item['New Tax'] for item in recommended) if recommended else tax_payable
    
    return render_template('results.html',
                         total_income=total_income,
                         assessable_income=assessable_income,
                         earned_income_relief=earned_income_relief,
                         parent_relief=parent_relief,
                         child_relief=child_relief,
                         working_mother_relief=working_mother_relief,
                         cpf_relief=cpf_relief,
                         srs_relief=srs_relief,
                         course_fee_relief=course_fee_relief,
                         nsman_relief=nsman_relief,
                         donation_relief=donation_relief,
                         total_reliefs=total_reliefs,
                         chargeable_income=chargeable_income,
                         tax_payable=tax_payable,
                         final_tax_payable=final_tax_payable,
                         recommended=recommended,
                         total_potential_savings=total_potential_savings,
                         new_tax=new_tax)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        message = request.form.get('message', '')
        response = get_chatbot_response(message)
        return jsonify({'response': response})
    return render_template('chatbot.html')

def get_chatbot_response(message):
    """
    Get a response from ChatGPT about tax relief schemes
    """
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a friendly and knowledgeable tax expert specializing in Singapore tax relief schemes. 
                Your role is to explain tax reliefs in simple, easy-to-understand terms. 
                Keep your explanations concise and practical, using everyday examples when possible.
                Focus on helping users understand how they can benefit from various reliefs."""},
                {"role": "user", "content": message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again later."

if __name__ == '__main__':
    app.run(debug=True)
