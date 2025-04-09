# Relief cap and tax bracket boundaries
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
            rebate = min(tax_payable * 0.6, 200)  # As part of the SG60 package, the rebate will be 60% of tax payable, capped at $200 per taxpayer
            tax_payable -= rebate
            break
        else:
            continue
    
    return tax_payable

# Example Usage
income = 150000
print(f"Tax payable for income {income}: ${calculate_tax(income):,.2f}")