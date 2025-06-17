import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime
import os
import uuid
import numpy as np

# ========== SETTINGS ==========
DATA_FILE = "finance_data.csv"
ARCHIVE_FILE = "finance_archive.csv"
LOANS_FILE = "loans_data.csv"
INVESTMENTS_FILE = "investments_data.csv"
BANKS = ["Commercial Bank", "DFCC Bank", "Sampath Bank", "BOC"]

# ========== PAGE SETUP ==========
st.set_page_config(
    page_title="Villa Finance Tracker",
    page_icon="🏡",
    layout="wide"
)
st.title("Finance Tracker")

# ========== DATA FUNCTIONS ==========
def load_or_create_data(file_path, columns):
    """Load existing data or create new DataFrame with proper type conversion"""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            # Ensure all expected columns exist
            for col in columns:
                if col not in df.columns:
                    df[col] = None if col != "ID" else str(uuid.uuid4())
            
            # Type conversions
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            if 'Amount' in df.columns:
                df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
            if 'IsLoan' in df.columns:
                df['IsLoan'] = df['IsLoan'].fillna(False)
            if 'IsInvestment' in df.columns:
                df['IsInvestment'] = df['IsInvestment'].fillna(False)
            
            return df[columns]  # Return only the expected columns
        except Exception as e:
            st.error(f"Error loading {file_path}: {str(e)}")
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data(df, file_path):
    """Save DataFrame to CSV with error handling"""
    try:
        df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving {file_path}: {str(e)}")
        return False

# ========== LOAD ALL DATA ==========
data_columns = ["ID", "Date", "Category", "Type", "Amount", "Note", "Bank", "Paid From Bank", "Purpose", "IsLoan", "IsInvestment"]
df = load_or_create_data(DATA_FILE, data_columns)
archive_df = load_or_create_data(ARCHIVE_FILE, data_columns)

loans_columns = ["ID", "Date", "Person", "Amount", "PaidBack", "Note", "SourceBank"]
loans_df = load_or_create_data(LOANS_FILE, loans_columns)
if not loans_df.empty:
    loans_df['Amount'] = pd.to_numeric(loans_df['Amount'], errors='coerce').fillna(0)
    loans_df['PaidBack'] = loans_df['PaidBack'].fillna(False)
    loans_df['SourceBank'] = loans_df['SourceBank'].fillna('')
    if 'Date' in loans_df.columns:
        loans_df['Date'] = pd.to_datetime(loans_df['Date'], errors='coerce').dt.date

investments_columns = ["ID", "Date", "Investment", "Amount", "Returned", "Note", "SourceBank"]
investments_df = load_or_create_data(INVESTMENTS_FILE, investments_columns)
if not investments_df.empty:
    investments_df['Amount'] = pd.to_numeric(investments_df['Amount'], errors='coerce').fillna(0)
    investments_df['Returned'] = investments_df['Returned'].fillna(False)
    investments_df['SourceBank'] = investments_df['SourceBank'].fillna('')
    if 'Date' in investments_df.columns:
        investments_df['Date'] = pd.to_datetime(investments_df['Date'], errors='coerce').dt.date

# ========== CALCULATION FUNCTIONS ==========
def calculate_bank_balances():
    """Calculate current balances for each bank with error checking"""
    if df.empty or not {'Type', 'Bank', 'Paid From Bank', 'Amount'}.issubset(df.columns):
        return {bank: 0 for bank in BANKS}
    
    balances = {bank: 0 for bank in BANKS}
    for _, row in df.iterrows():
        try:
            if row["Type"] == "Income" and row["Bank"] in BANKS:
                balances[row["Bank"]] += float(row["Amount"])
            elif row["Type"] == "Expense" and row["Paid From Bank"] in BANKS and not row["IsLoan"]:
                balances[row["Paid From Bank"]] -= float(row["Amount"])
        except (ValueError, TypeError):
            continue
    return balances

def calculate_category_balances():
    """Calculate balances for each category with error checking"""
    if df.empty:
        return {
            "villa_balance": 0,
            "personal_balance": 0,
            "villa_income": 0,
            "villa_expense": 0,
            "personal_income": 0,
            "personal_expense": 0
        }
    
    try:
        # Exclude loan transactions from personal expenses
        villa_income = df[(df["Category"] == "Villa") & (df["Type"] == "Income")]["Amount"].sum()
        villa_expense = df[(df["Category"] == "Villa") & (df["Type"] == "Expense")]["Amount"].sum()
        personal_income = df[(df["Category"] == "Personal") & (df["Type"] == "Income")]["Amount"].sum()
        personal_expense = df[(df["Category"] == "Personal") & (df["Type"] == "Expense") & (~df["IsLoan"])]["Amount"].sum()
    except:
        villa_income = villa_expense = personal_income = personal_expense = 0
    
    return {
        "villa_balance": villa_income - villa_expense,
        "personal_balance": personal_income - personal_expense,
        "villa_income": villa_income,
        "villa_expense": villa_expense,
        "personal_income": personal_income,
        "personal_expense": personal_expense
    }

# ========== DELETE TRANSACTION FUNCTION ==========
def delete_transaction(transaction_id):
    """Delete a transaction and update all related records"""
    global df, archive_df, loans_df, investments_df
    
    try:
        # Find the transaction to delete
        transaction = df[df['ID'] == transaction_id].iloc[0]
        
        # Add to archive before deleting
        archive_df = pd.concat([archive_df, pd.DataFrame([transaction])], ignore_index=True)
        save_data(archive_df, ARCHIVE_FILE)
        
        # Remove from main dataframe
        df = df[df['ID'] != transaction_id]
        
        # If it's a loan-related transaction, update loans records
        if transaction['IsLoan']:
            if transaction['Type'] == 'Expense':  # Original loan
                # Find matching loan record
                loan_note = f"Loan to {transaction['Note'].split('to ')[-1]}" if 'to ' in transaction['Note'] else transaction['Note']
                matching_loans = loans_df[
                    (loans_df['Person'].str.contains(loan_note.split('to ')[-1][:20])) & 
                    (loans_df['Amount'] == transaction['Amount']) & 
                    (loans_df['SourceBank'] == transaction['Paid From Bank'])
                ]
                
                if not matching_loans.empty:
                    loans_df = loans_df[~loans_df['ID'].isin(matching_loans['ID'])]
                    save_data(loans_df, LOANS_FILE)
            
            elif transaction['Type'] == 'Income':  # Loan repayment
                # Find matching loan record
                loan_note = f"Loan repayment from {transaction['Note'].split('from ')[-1]}" if 'from ' in transaction['Note'] else transaction['Note']
                matching_loans = loans_df[
                    (loans_df['Person'].str.contains(loan_note.split('from ')[-1][:20])) & 
                    (loans_df['Amount'] <= transaction['Amount']) & 
                    (~loans_df['PaidBack'])
                ]
                
                if not matching_loans.empty:
                    # Restore the loan amount
                    loans_df.loc[loans_df['ID'].isin(matching_loans['ID']), 'Amount'] += transaction['Amount']
                    loans_df.loc[loans_df['ID'].isin(matching_loans['ID']), 'PaidBack'] = False
                    save_data(loans_df, LOANS_FILE)
        
        # If it's an investment-related transaction, update investments records
        elif transaction['IsInvestment']:
            if transaction['Type'] == 'Expense':  # Original investment
                # Find matching investment record
                invest_note = f"Investment in {transaction['Note'].split('in ')[-1]}" if 'in ' in transaction['Note'] else transaction['Note']
                matching_investments = investments_df[
                    (investments_df['Investment'].str.contains(invest_note.split('in ')[-1][:20])) & 
                    (investments_df['Amount'] == transaction['Amount']) & 
                    (investments_df['SourceBank'] == transaction['Paid From Bank'])
                ]
                
                if not matching_investments.empty:
                    investments_df = investments_df[~investments_df['ID'].isin(matching_investments['ID'])]
                    save_data(investments_df, INVESTMENTS_FILE)
            
            elif transaction['Type'] == 'Income':  # Investment return
                # Find matching investment record
                invest_note = f"Return from {transaction['Note'].split('from ')[-1]}" if 'from ' in transaction['Note'] else transaction['Note']
                matching_investments = investments_df[
                    (investments_df['Investment'].str.contains(invest_note.split('from ')[-1][:20])) & 
                    (investments_df['Amount'] <= transaction['Amount']) & 
                    (~investments_df['Returned'])
                ]
                
                if not matching_investments.empty:
                    # Restore the investment amount
                    investments_df.loc[investments_df['ID'].isin(matching_investments['ID']), 'Amount'] += transaction['Amount']
                    investments_df.loc[investments_df['ID'].isin(matching_investments['ID']), 'Returned'] = False
                    save_data(investments_df, INVESTMENTS_FILE)
        
        # Save the main dataframe
        if save_data(df, DATA_FILE):
            st.success("Transaction deleted and records updated!")
            st.rerun()
    
    except Exception as e:
        st.error(f"Error deleting transaction: {str(e)}")

# ========== FINANCIAL OVERVIEW ==========
st.subheader("💰 Financial Overview")
bank_balances = calculate_bank_balances()
category_balances = calculate_category_balances()

# Calculate totals with error handling
try:
    total_bank_balance = sum(bank_balances.values())
    total_loans = loans_df[~loans_df['PaidBack']]['Amount'].sum() if not loans_df.empty and 'PaidBack' in loans_df.columns else 0
    total_investments = investments_df[~investments_df['Returned']]['Amount'].sum() if not investments_df.empty and 'Returned' in investments_df.columns else 0
    net_worth = total_bank_balance + total_loans + total_investments
except:
    total_bank_balance = total_loans = total_investments = net_worth = 0

cols = st.columns(4)
cols[0].metric("Total Bank Balance", f"Rs. {total_bank_balance:,.2f}")
cols[1].metric("Active Loans", f"Rs. {total_loans:,.2f}")
cols[2].metric("Active Investments", f"Rs. {total_investments:,.2f}")
cols[3].metric("Net Worth", f"Rs. {net_worth:,.2f}")

# ========== BALANCE SECTIONS ==========
st.subheader("📊 Category Balances")
balance_cols = st.columns(2)

with balance_cols[0]:
    st.metric("🏡 Villa Balance", 
             f"Rs. {category_balances['villa_balance']:,.2f}",
             delta=f"Income: Rs. {category_balances['villa_income']:,.2f} | Expenses: Rs. {category_balances['villa_expense']:,.2f}",
             delta_color="normal")

with balance_cols[1]:
    st.metric("👤 Personal Balance", 
             f"Rs. {category_balances['personal_balance']:,.2f}",
             delta=f"Income: Rs. {category_balances['personal_income']:,.2f} | Expenses: Rs. {category_balances['personal_expense']:,.2f}",
             delta_color="normal")

st.subheader("🏦 Bank Balances")
bank_cols = st.columns(len(BANKS))
for col, bank in zip(bank_cols, BANKS):
    col.metric(bank, f"Rs. {bank_balances[bank]:,.2f}")

# ========== LOANS SECTION ==========
st.divider()
st.subheader("📝 Loans Management")

with st.expander("➕ Add New Loan", expanded=True):
    with st.form("loan_form", clear_on_submit=True):
        loan_date = st.date_input("Loan Date", date.today())
        person = st.text_input("Person/Organization", max_chars=100)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0, format="%.2f")
        source_bank = st.selectbox("Paid From Bank", BANKS + ["Cash"], index=0)
        loan_note = st.text_input("Loan Purpose", max_chars=200)
        
        if st.form_submit_button("Record Loan"):
            try:
                new_loan = {
                    "ID": str(uuid.uuid4()),
                    "Date": loan_date,
                    "Person": person[:100],  # Ensure length limit
                    "Amount": float(loan_amount),
                    "PaidBack": False,
                    "Note": loan_note[:200],
                    "SourceBank": source_bank
                }
                loans_df = pd.concat([loans_df, pd.DataFrame([new_loan])], ignore_index=True)
                if save_data(loans_df, LOANS_FILE):
                    if source_bank in BANKS:
                        new_trans = {
                            "ID": str(uuid.uuid4()),
                            "Date": loan_date,
                            "Category": "Personal",
                            "Type": "Expense",
                            "Amount": float(loan_amount),
                            "Note": f"Loan to {person[:50]}",
                            "Bank": "",
                            "Paid From Bank": source_bank,
                            "Purpose": loan_note[:100],
                            "IsLoan": True,
                            "IsInvestment": False
                        }
                        df = pd.concat([df, pd.DataFrame([new_trans])], ignore_index=True)
                        save_data(df, DATA_FILE)
                    st.success("Loan recorded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error recording loan: {str(e)}")

# Display active loans
st.write("### Active Loans")
if not loans_df.empty and 'PaidBack' in loans_df.columns:
    active_loans = loans_df[~loans_df['PaidBack']].copy()
    if not active_loans.empty:
        for _, loan in active_loans.iterrows():
            cols = st.columns([1, 2, 2, 2, 3, 2])
            cols[0].write(loan['Date'].strftime('%Y-%m-%d') if pd.notna(loan['Date']) else "No date")
            cols[1].write(loan['Person'][:30] + '...' if len(str(loan['Person'])) > 30 else loan['Person'])
            cols[2].write(f"Rs. {float(loan['Amount']):,.2f}")
            cols[3].write(loan['SourceBank'][:20])
            
            with cols[4]:
                repay_amount = st.number_input(
                    "Repayment Amount", 
                    min_value=0.0, 
                    max_value=float(loan['Amount']),
                    step=100.0,
                    format="%.2f",
                    key=f"repay_{loan['ID']}"
                )
                deposit_bank = st.selectbox(
                    "Deposit to Bank", 
                    BANKS, 
                    key=f"deposit_bank_{loan['ID']}"
                )
            
            with cols[5]:
                if st.button("Record Repayment", key=f"repay_btn_{loan['ID']}"):
                    try:
                        if repay_amount > 0:
                            # Record transaction
                            new_trans = {
                                "ID": str(uuid.uuid4()),
                                "Date": date.today(),
                                "Category": "Personal",
                                "Type": "Income",
                                "Amount": float(repay_amount),
                                "Note": f"Loan repayment from {loan['Person'][:50]}",
                                "Bank": deposit_bank,
                                "Paid From Bank": "",
                                "Purpose": "",
                                "IsLoan": True,
                                "IsInvestment": False
                            }
                            df = pd.concat([df, pd.DataFrame([new_trans])], ignore_index=True)
                            
                            # Update loan amount
                            loans_df.loc[loans_df['ID'] == loan['ID'], 'Amount'] -= float(repay_amount)
                            
                            # Mark as paid back if amount is zero
                            if abs(loans_df.loc[loans_df['ID'] == loan['ID'], 'Amount'].values[0]) < 0.01:
                                loans_df.loc[loans_df['ID'] == loan['ID'], 'PaidBack'] = True
                                loans_df.loc[loans_df['ID'] == loan['ID'], 'Amount'] = 0
                            
                            if save_data(df, DATA_FILE) and save_data(loans_df, LOANS_FILE):
                                st.success("Repayment recorded!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error recording repayment: {str(e)}")
    else:
        st.info("No active loans")
else:
    st.info("No loans recorded yet")

# ========== INVESTMENTS SECTION ==========
st.divider()
st.subheader("💼 Investments Management")

with st.expander("➕ Add New Investment", expanded=True):
    with st.form("investment_form", clear_on_submit=True):
        invest_date = st.date_input("Investment Date", date.today())
        investment_name = st.text_input("Investment Name", max_chars=100)
        invest_amount = st.number_input("Investment Amount", min_value=0.0, step=100.0, format="%.2f")
        source_bank = st.selectbox("Paid From Bank", BANKS + ["Cash"], index=0)
        invest_note = st.text_input("Investment Details", max_chars=200)
        
        if st.form_submit_button("Record Investment"):
            try:
                new_investment = {
                    "ID": str(uuid.uuid4()),
                    "Date": invest_date,
                    "Investment": investment_name[:100],
                    "Amount": float(invest_amount),
                    "Returned": False,
                    "Note": invest_note[:200],
                    "SourceBank": source_bank
                }
                investments_df = pd.concat([investments_df, pd.DataFrame([new_investment])], ignore_index=True)
                if save_data(investments_df, INVESTMENTS_FILE):
                    if source_bank in BANKS:
                        new_trans = {
                            "ID": str(uuid.uuid4()),
                            "Date": invest_date,
                            "Category": "Personal",
                            "Type": "Expense",
                            "Amount": float(invest_amount),
                            "Note": f"Investment in {investment_name[:50]}",
                            "Bank": "",
                            "Paid From Bank": source_bank,
                            "Purpose": invest_note[:100],
                            "IsLoan": False,
                            "IsInvestment": True
                        }
                        df = pd.concat([df, pd.DataFrame([new_trans])], ignore_index=True)
                        save_data(df, DATA_FILE)
                    st.success("Investment recorded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error recording investment: {str(e)}")

# Display active investments
st.write("### Active Investments")
if not investments_df.empty and 'Returned' in investments_df.columns:
    active_investments = investments_df[~investments_df['Returned']].copy()
    if not active_investments.empty:
        for _, investment in active_investments.iterrows():
            cols = st.columns([1, 2, 2, 2, 3, 2])
            cols[0].write(investment['Date'].strftime('%Y-%m-%d') if pd.notna(investment['Date']) else "No date")
            cols[1].write(investment['Investment'][:30] + '...' if len(str(investment['Investment'])) > 30 else investment['Investment'])
            cols[2].write(f"Rs. {float(investment['Amount']):,.2f}")
            cols[3].write(investment['SourceBank'][:20])
            
            with cols[4]:
                return_amount = st.number_input(
                    "Return Amount", 
                    min_value=0.0, 
                    max_value=float(investment['Amount']),
                    step=100.0,
                    format="%.2f",
                    key=f"return_{investment['ID']}"
                )
                deposit_bank = st.selectbox(
                    "Deposit to Bank", 
                    BANKS, 
                    key=f"return_bank_{investment['ID']}"
                )
            
            with cols[5]:
                if st.button("Record Return", key=f"return_btn_{investment['ID']}"):
                    try:
                        if return_amount > 0:
                            # Record transaction
                            new_trans = {
                                "ID": str(uuid.uuid4()),
                                "Date": date.today(),
                                "Category": "Personal",
                                "Type": "Income",
                                "Amount": float(return_amount),
                                "Note": f"Return from {investment['Investment'][:50]}",
                                "Bank": deposit_bank,
                                "Paid From Bank": "",
                                "Purpose": "",
                                "IsLoan": False,
                                "IsInvestment": True
                            }
                            df = pd.concat([df, pd.DataFrame([new_trans])], ignore_index=True)
                            
                            # Update investment amount
                            investments_df.loc[investments_df['ID'] == investment['ID'], 'Amount'] -= float(return_amount)
                            
                            # Mark as returned if amount is zero
                            if abs(investments_df.loc[investments_df['ID'] == investment['ID'], 'Amount'].values[0]) < 0.01:
                                investments_df.loc[investments_df['ID'] == investment['ID'], 'Returned'] = True
                                investments_df.loc[investments_df['ID'] == investment['ID'], 'Amount'] = 0
                            
                            if save_data(df, DATA_FILE) and save_data(investments_df, INVESTMENTS_FILE):
                                st.success("Investment return recorded!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error recording return: {str(e)}")
    else:
        st.info("No active investments")
else:
    st.info("No investments recorded yet")

# ========== FINANCIAL REPORTS ==========
st.divider()
st.subheader("📊 Financial Reports")

# Date selection
col1, col2 = st.columns(2)
start_date = col1.date_input("From Date", value=date.today().replace(day=1))
end_date = col2.date_input("To Date", value=date.today())

if st.button("Generate Report", type="primary"):
    if df.empty or 'Date' not in df.columns:
        st.warning("No transaction data available")
    else:
        try:
            # Convert dates and filter
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            period_data = df[mask].copy()
            
            if period_data.empty:
                st.warning("No transactions found for selected period")
            else:
                # Calculate report values
                villa_expense = period_data[(period_data["Category"] == "Villa") & 
                                          (period_data["Type"] == "Expense")]["Amount"].sum()
                personal_expense = period_data[(period_data["Category"] == "Personal") & 
                                             (period_data["Type"] == "Expense") & 
                                             (~period_data["IsLoan"])]["Amount"].sum()
                villa_income = period_data[(period_data["Category"] == "Villa") & 
                                         (period_data["Type"] == "Income")]["Amount"].sum()
                personal_income = period_data[(period_data["Category"] == "Personal") & 
                                            (period_data["Type"] == "Income")]["Amount"].sum()
                
                villa_balance = villa_income - villa_expense
                personal_balance = personal_income - personal_expense
                
                # Display report
                st.subheader(f"📈 Financial Report: {start_date} to {end_date}")
                
                # Summary metrics
                cols = st.columns(4)
                cols[0].metric("Villa Income", f"Rs. {villa_income:,.2f}", delta_color="off")
                cols[1].metric("Personal Income", f"Rs. {personal_income:,.2f}", delta_color="off")
                cols[2].metric("Villa Expenses", f"Rs. {villa_expense:,.2f}", delta_color="inverse")
                cols[3].metric("Personal Expenses", f"Rs. {personal_expense:,.2f}", delta_color="inverse")
                
                # Category balances
                st.subheader("Category Balances for Period")
                balance_cols = st.columns(2)
                
                with balance_cols[0]:
                    st.metric("🏡 Villa Balance", 
                             f"Rs. {villa_balance:,.2f}",
                             delta=f"Income: Rs. {villa_income:,.2f} | Expenses: Rs. {villa_expense:,.2f}",
                             delta_color="normal")
                
                with balance_cols[1]:
                    st.metric("👤 Personal Balance", 
                             f"Rs. {personal_balance:,.2f}",
                             delta=f"Income: Rs. {personal_income:,.2f} | Expenses: Rs. {personal_expense:,.2f}",
                             delta_color="normal")
                
                # Create trend data for the chart
                period_data['DateOnly'] = period_data['Date'].dt.date
                
                period_data['Villa Income'] = np.where(
                    (period_data['Category'] == 'Villa') & (period_data['Type'] == 'Income'),
                    period_data['Amount'], 0
                )
                
                period_data['Personal Income'] = np.where(
                    (period_data['Category'] == 'Personal') & (period_data['Type'] == 'Income'),
                    period_data['Amount'], 0
                )
                
                period_data['Villa Expenses'] = np.where(
                    (period_data['Category'] == 'Villa') & (period_data['Type'] == 'Expense'),
                    period_data['Amount'], 0
                )
                
                period_data['Personal Expenses'] = np.where(
                    (period_data['Category'] == 'Personal') & (period_data['Type'] == 'Expense') & (~period_data['IsLoan']),
                    period_data['Amount'], 0
                )
                
                # Group by date and sum
                daily_data = period_data.groupby('DateOnly').agg({
                    'Villa Income': 'sum',
                    'Personal Income': 'sum',
                    'Villa Expenses': 'sum',
                    'Personal Expenses': 'sum'
                }).reset_index().sort_values('DateOnly')
                
                # Create the line chart
                fig = px.line(daily_data, 
                             x='DateOnly', 
                             y=['Villa Income', 'Personal Income', 'Villa Expenses', 'Personal Expenses'],
                             title="Income and Expense Trends",
                             labels={'value': 'Amount (Rs.)', 'DateOnly': 'Date', 'variable': 'Category'},
                             color_discrete_map={
                                 'Villa Income': 'darkgreen',
                                 'Personal Income': 'gold',
                                 'Villa Expenses': 'pink',
                                 'Personal Expenses': 'red'
                             })
                
                fig.update_layout(
                    hovermode="x unified",
                    legend_title_text='Category'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                try:
                    csv = period_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Full Report as CSV",
                        data=csv,
                        file_name=f"finance_report_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

# ========== TRANSACTION MANAGEMENT ==========
st.divider()
st.subheader("💵 Transaction Management")

# Add new transaction
with st.expander("➕ Add New Transaction", expanded=False):
    with st.form("new_transaction", clear_on_submit=True):
        trans_date = st.date_input("Transaction Date", date.today())
        category = st.selectbox("Category", ["Villa", "Personal"])
        trans_type = st.selectbox("Type", ["Income", "Expense"])
        amount = st.number_input("Amount (LKR)", min_value=0.0, step=0.01, format="%.2f")
        note = st.text_input("Description", max_chars=200)
        
        if trans_type == "Income":
            bank = st.selectbox("Deposited to Bank", BANKS)
            paid_from = ""
            purpose = ""
        else:
            bank = ""
            paid_from = st.selectbox("Paid From Bank", BANKS)
            purpose = st.text_input("Payment Purpose", max_chars=200)
        
        if st.form_submit_button("Save Transaction"):
            try:
                new_trans = {
                    "ID": str(uuid.uuid4()),
                    "Date": trans_date,
                    "Category": category,
                    "Type": trans_type,
                    "Amount": float(amount),
                    "Note": note[:200],
                    "Bank": bank,
                    "Paid From Bank": paid_from,
                    "Purpose": purpose[:200],
                    "IsLoan": False,
                    "IsInvestment": False
                }
                df = pd.concat([df, pd.DataFrame([new_trans])], ignore_index=True)
                if save_data(df, DATA_FILE):
                    st.success("Transaction saved successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error saving transaction: {str(e)}")

# Display transactions with delete option
st.write("### Recent Transactions")
if not df.empty:
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        recent_df = df.sort_values("Date", ascending=False).head(20).copy()
        recent_df['Date'] = recent_df['Date'].dt.date
        
        # Display each transaction with a delete button
        for _, row in recent_df.iterrows():
            cols = st.columns([1, 1, 1, 2, 2, 2, 2, 1])
            cols[0].write(row['Date'].strftime('%Y-%m-%d') if pd.notna(row['Date']) else "No date")
            cols[1].write(row['Category'])
            cols[2].write(row['Type'])
            cols[3].write(f"Rs. {float(row['Amount']):,.2f}")
            cols[4].write(row['Note'][:30] + '...' if len(str(row['Note'])) > 30 else row['Note'])
            cols[5].write(row['Bank'] if row['Bank'] else row['Paid From Bank'])
            cols[6].write(row['Purpose'][:30] + '...' if len(str(row['Purpose'])) > 30 else row['Purpose'])
            
            with cols[7]:
                if st.button("🗑️", key=f"del_{row['ID']}"):
                    delete_transaction(row['ID'])
    except Exception as e:
        st.error(f"Error displaying transactions: {str(e)}")
else:
    st.info("No transactions recorded yet")

# ========== ARCHIVED TRANSACTIONS ==========
st.divider()
st.subheader("🗃️ Archived Transactions")

if not archive_df.empty:
    try:
        archive_df['Date'] = pd.to_datetime(archive_df['Date'], errors='coerce')
        st.dataframe(archive_df.sort_values("Date", ascending=False).head(20)[['Date', 'Category', 'Type', 'Amount', 'Note', 'Bank', 'Paid From Bank', 'Purpose']].rename(
            columns={
                'Paid From Bank': 'From Bank',
                'Purpose': 'Details'
            }
        ).style.format({'Amount': '{:,.2f}'}), height=400)
    except Exception as e:
        st.error(f"Error displaying archived transactions: {str(e)}")
else:
    st.info("No archived transactions yet")

# ========== NET WORTH TREND CHART ==========
st.divider()
st.subheader("📈 Net Worth Trend")

if not df.empty:
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Calculate daily changes for bank, loans, and investments
        df['BankChange'] = np.where(
            df['Type'] == 'Income', 
            df['Amount'], 
            np.where(
                df['Type'] == 'Expense',
                -df['Amount'],
                0
            )
        )
        
        df['LoanChange'] = np.where(
            df['IsLoan'] & (df['Type'] == 'Income'),
            df['Amount'],
            np.where(
                df['IsLoan'] & (df['Type'] == 'Expense'),
                -df['Amount'],
                0
            )
        )
        
        df['InvestmentChange'] = np.where(
            df['IsInvestment'] & (df['Type'] == 'Income'),
            df['Amount'],
            np.where(
                df['IsInvestment'] & (df['Type'] == 'Expense'),
                -df['Amount'],
                0
            )
        )
        
        # Group by date and calculate cumulative sums
        daily_data = df.groupby(df['Date'].dt.date).agg({
            'BankChange': 'sum',
            'LoanChange': 'sum',
            'InvestmentChange': 'sum'
        }).reset_index().sort_values('Date')
        
        daily_data['Net Worth'] = daily_data[['BankChange', 'LoanChange', 'InvestmentChange']].cumsum().sum(axis=1)
        
        # Create the line chart
        fig = px.line(daily_data, 
                     x='Date', 
                     y='Net Worth',
                     title="Net Worth Over Time",
                     labels={'Net Worth': 'Amount (Rs.)'})
        fig.update_traces(line=dict(color='blue', width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Net Worth Calculation:**
        - Total Bank Balance: Rs. {total_bank_balance:,.2f}
        + Active Loans: Rs. {total_loans:,.2f}
        + Active Investments: Rs. {total_investments:,.2f}
        = **Total Net Worth: Rs. {net_worth:,.2f}**
        """)
    except Exception as e:
        st.error(f"Error generating net worth trend: {str(e)}")
else:
    st.info("No transaction data available for trend analysis")

# ========== FOOTER ==========
st.divider()
st.caption("© 2023 Villa by 11.11 - Finance Tracking System")