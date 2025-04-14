import math

# Tax brackets and relief caps
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

def calculate_tax(income):
    """
    Calculate tax based on Singapore's progressive tax rates.
    Includes the 60% rebate capped at $200.
    """
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
            tax_payable = tax_brackets[i][2] + (remaining_income * (tax_brackets[i+1][1] / 100))
            rebate = min(tax_payable * 0.6, 200)  # 60% rebate capped at $200
            tax_payable -= rebate
            break
        else:
            continue
    
    return max(0, tax_payable)

def calculate_earned_income_relief(age, taxable_earned_income):
    """
    Calculate Earned Income Relief based on age and taxable earned income.
    """
    if age < 55:
        return min(1000, taxable_earned_income)
    elif 55 <= age < 60:
        return min(6000, taxable_earned_income)
    else:  # 60 and above
        return min(8000, taxable_earned_income)

def calculate_parent_relief(parent_type, num_dependents, living_with_you):
    """
    Calculate Parent Relief based on type, number of dependents, and living arrangement.
    """
    relief = 0
    if parent_type == "parent":
        if living_with_you:
            relief = 9000 * min(2, num_dependents)
        else:
            relief = 5500 * min(2, num_dependents)
    elif parent_type == "handicapped":
        if living_with_you:
            relief = 14000 * min(2, num_dependents)
        else:
            relief = 10000 * min(2, num_dependents)
    return relief

def calculate_child_relief(num_children, num_handicapped_children):
    """
    Calculate Child Relief based on number of children and handicapped children.
    """
    return (4000 * num_children) + (7500 * num_handicapped_children)

def calculate_working_mother_child_relief(num_children):
    """
    Calculate Working Mother's Child Relief based on number of children.
    """
    relief = 0
    if num_children == 1:
        relief = 15000
    elif num_children == 2:
        relief = 30000
    elif num_children >= 3:
        relief = 45000
    return relief

def calculate_cpf_relief(contributions):
    """
    Calculate CPF Relief based on contributions.
    """
    return min(8000, contributions)

def calculate_srs_relief(contributions, is_singaporean=True):
    """
    Calculate SRS Relief based on contributions and citizenship status.
    """
    if is_singaporean:
        return min(15300, contributions)
    else:
        return min(35700, contributions)

def calculate_donation_relief(donation_amount):
    """
    Calculate Donation Relief (2.5 times the donation amount).
    """
    return 2.5 * donation_amount

def calculate_course_fee_relief(course_fees):
    """
    Calculate Course Fee Relief.
    """
    return min(5500, course_fees)

def calculate_nsman_relief(ns_type, performed_ns, is_key_appointment):
    """
    Calculate NSman Relief based on type and conditions.
    """
    if ns_type == "self":
        if performed_ns:
            if is_key_appointment:
                return 5000
            return 3000
        else:
            if is_key_appointment:
                return 3500
            return 1500
    elif ns_type == "wife":
        return 750
    elif ns_type == "parent":
        return 750
    return 0

def calculate_parenthood_tax_rebate(children_data):
    """
    Calculate Parenthood Tax Rebate based on children's birth years and order.
    """
    rebate = 0
    for child in children_data:
        birth_year = child['birth_year']
        birth_order = child['birth_order']
        
        if birth_year >= 2008:
            if birth_order == 1:
                rebate += 5000
            elif birth_order in [2, 3, 4]:
                rebate += 10000
            elif birth_order >= 5:
                rebate += 20000
        else:  # 2004-2007
            if birth_order == 2:
                rebate += 10000
            elif birth_order in [3, 4]:
                rebate += 20000
    
    return rebate 