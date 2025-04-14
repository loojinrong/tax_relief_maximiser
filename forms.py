from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SelectField, BooleanField, FloatField
from wtforms.validators import DataRequired, NumberRange, Optional, InputRequired

class GeneralInfoForm(FlaskForm):
    citizenship = SelectField('Are you a Singapore Citizen or PR or a Foreigner?',
                            choices=[('citizen', 'Singapore Citizen'),
                                   ('pr', 'Permanent Resident'),
                                   ('foreigner', 'Foreigner')],
                            validators=[DataRequired()])
    gender = SelectField('What is your gender?',
                        choices=[('male', 'Male'),
                               ('female', 'Female')],
                        validators=[DataRequired()])
    marital_status = SelectField('What is your marital status?',
                               choices=[('single', 'Single'),
                                      ('married', 'Married'),
                                      ('divorced', 'Divorced'),
                                      ('widowed', 'Widowed')],
                               validators=[DataRequired()])
    age_2025 = IntegerField('What is your age in 2025?',
                          validators=[InputRequired(), NumberRange(min=0)])
    disability = SelectField('Do you have any disability?',
                           choices=[('yes', 'Yes'),
                                  ('no', 'No')],
                           validators=[DataRequired()])
    num_children = IntegerField('How many children do you have as of 2025?',
                              validators=[InputRequired(), NumberRange(min=0)])
    num_handicapped_children = IntegerField('How many handicapped children do you have as of 2025?',
                                          validators=[InputRequired(), NumberRange(min=0)])

class IncomeForm(FlaskForm):
    employment_income = FloatField('Employment Income',
                                 validators=[InputRequired(), NumberRange(min=0)])
    employment_expenses = FloatField('Employment Expenses',
                                   validators=[InputRequired(), NumberRange(min=0)])
    trade_business_income = FloatField('Trade, Business, Profession, and Vocation Income',
                                     validators=[InputRequired(), NumberRange(min=0)])
    dividends = FloatField('Dividends',
                         validators=[InputRequired(), NumberRange(min=0)])
    interest = FloatField('Interest',
                        validators=[InputRequired(), NumberRange(min=0)])
    rent = FloatField('Net Rent from Property',
                     validators=[InputRequired(), NumberRange(min=0)])
    approved_donations = FloatField('Approved Donations',
                                  validators=[InputRequired(), NumberRange(min=0)])

class ReliefForm(FlaskForm):
    parent_relief_type = SelectField('Parent Relief Type',
                                   choices=[('none', 'No'),
                                          ('parent', 'Parent Relief'),
                                          ('handicapped', 'Handicapped Parent Relief'),
                                          ('both', 'Both')],
                                   validators=[DataRequired()])
    num_parent_dependents = IntegerField('Number of Parent Dependents',
                                       validators=[InputRequired(), NumberRange(min=0, max=2)])
    parent_living_with_you = SelectField('Are your parents living with you?',
                                       choices=[('yes', 'Yes'),
                                              ('no', 'No')],
                                       validators=[Optional()])
    cpf_contributions = FloatField('CPF Contributions',
                                 validators=[InputRequired(), NumberRange(min=0)])
    self_top_up_cpf = FloatField('Self Top-Up CPF',
                               validators=[InputRequired(), NumberRange(min=0)])
    srs_contributions = FloatField('SRS Contributions',
                                 validators=[InputRequired(), NumberRange(min=0)])
    course_fees = FloatField('Course Fees',
                           validators=[InputRequired(), NumberRange(min=0)])
    nsman_type = SelectField('NSman Type',
                           choices=[('none', 'Not Applicable'),
                                  ('self', 'Self'),
                                  ('wife', 'Wife'),
                                  ('parent', 'Parent')],
                           validators=[DataRequired()])
    performed_ns = SelectField('Did you perform NS activities?',
                             choices=[('yes', 'Yes'),
                                    ('no', 'No')],
                             validators=[Optional()])
    is_key_appointment = SelectField('Are you a NS key command and staff appointment holder?',
                                   choices=[('yes', 'Yes'),
                                          ('no', 'No')],
                                   validators=[Optional()])

class ChildrenInfoForm(FlaskForm):
    def __init__(self, num_children=0, *args, **kwargs):
        super(ChildrenInfoForm, self).__init__(*args, **kwargs)
        self.children = []
        for i in range(num_children):
            child_form = ChildForm(prefix=f'child_{i}')
            self.children.append(child_form)

class ChildForm(FlaskForm):
    birth_year = IntegerField('Birth Year', validators=[InputRequired(), NumberRange(min=1900, max=2025)])
    birth_order = IntegerField('Birth Order', validators=[InputRequired(), NumberRange(min=1)])
    is_handicapped = BooleanField('Is Handicapped?')

def calculate_tax_reliefs(form_data):
    # Initialize all reliefs to 0
    earned_income_relief = 0
    parent_relief = 0
    child_relief = 0
    working_mother_relief = 0
    cpf_relief = 0
    srs_relief = 0
    course_fee_relief = 0
    nsman_relief = 0
    
    # Calculate earned income relief
    employment_income = form_data.get('employment_income', 0)
    if employment_income > 0:
        earned_income_relief = min(1000, employment_income)
    
    # Calculate parent relief
    if form_data.get('parent_relief_type') != 'none':
        num_parents = form_data.get('num_parent_dependents', 0)
        if form_data.get('parent_relief_type') == 'handicapped':
            parent_relief = num_parents * 8000
        else:
            parent_relief = num_parents * 5500
    
    # Calculate child relief
    num_children = form_data.get('num_children', 0)
    num_handicapped_children = form_data.get('num_handicapped_children', 0)
    child_relief = (num_children - num_handicapped_children) * 4000 + num_handicapped_children * 7500
    
    # Calculate working mother relief (simplified)
    if form_data.get('citizenship') in ['citizen', 'pr']:
        working_mother_relief = num_children * 1000
    
    # Calculate CPF relief
    cpf_relief = min(form_data.get('cpf_contributions', 0), 8000)
    
    # Calculate SRS relief
    srs_relief = min(form_data.get('srs_contributions', 0), 15300)
    
    # Calculate course fee relief
    course_fee_relief = min(form_data.get('course_fees', 0), 5500)
    
    # Calculate NSman relief
    if form_data.get('performed_ns') == 'yes':
        nsman_relief = 3000
        if form_data.get('is_key_appointment') == 'yes':
            nsman_relief += 2000
    
    return {
        'earned_income_relief': earned_income_relief,
        'parent_relief': parent_relief,
        'child_relief': child_relief,
        'working_mother_relief': working_mother_relief,
        'cpf_relief': cpf_relief,
        'srs_relief': srs_relief,
        'course_fee_relief': course_fee_relief,
        'nsman_relief': nsman_relief
    }

def calculate_chargeable_income(form_data):
    # Calculate total income
    total_income = (
        form_data.get('employment_income', 0) +
        form_data.get('trade_business_income', 0) +
        form_data.get('dividends', 0) +
        form_data.get('interest', 0) +
        form_data.get('rent', 0)
    )
    
    # Calculate total reliefs
    reliefs = calculate_tax_reliefs(form_data)
    total_reliefs = sum(reliefs.values())
    
    # Calculate chargeable income
    chargeable_income = max(0, total_income - total_reliefs)
    return chargeable_income

def calculate_tax(chargeable_income):
    # Singapore progressive tax rates for 2025
    if chargeable_income <= 20000:
        return chargeable_income * 0
    elif chargeable_income <= 30000:
        return (chargeable_income - 20000) * 0.02
    elif chargeable_income <= 40000:
        return 200 + (chargeable_income - 30000) * 0.035
    elif chargeable_income <= 80000:
        return 550 + (chargeable_income - 40000) * 0.07
    elif chargeable_income <= 120000:
        return 3350 + (chargeable_income - 80000) * 0.115
    elif chargeable_income <= 160000:
        return 7950 + (chargeable_income - 120000) * 0.15
    elif chargeable_income <= 200000:
        return 13950 + (chargeable_income - 160000) * 0.18
    elif chargeable_income <= 240000:
        return 21150 + (chargeable_income - 200000) * 0.19
    elif chargeable_income <= 280000:
        return 28750 + (chargeable_income - 240000) * 0.195
    elif chargeable_income <= 320000:
        return 36550 + (chargeable_income - 280000) * 0.20
    else:
        return 44550 + (chargeable_income - 320000) * 0.22

# Remove the undefined variables and replace with function calls
def calculate_total_tax(form_data):
    chargeable_income = calculate_chargeable_income(form_data)
    tax_payable = calculate_tax(chargeable_income)
    return tax_payable 