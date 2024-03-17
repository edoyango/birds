from send_email import send_email
import pandas as pd

def parse_csv_and_send(sender, pwd, csv_file, subject, body_template):

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        msg = body_template.format(
            FIRST=row["FIRST"],
            LAST=row["LAST"])
        send_email(sender, pwd, row["EMAIL"], subject, msg)

if __name__=="__main__":

    import argparse, os

    parser = argparse.ArgumentParser(
        prog = "send_email_from_csv.py",
        description = "a simple CLI utility to send an email using gmail SMTP servers, from a CSV file.")
    
    parser.add_argument("sender", help="Email address to send email from.")
    parser.add_argument("CSV", help="CSV file to pull recipients from.")
    parser.add_argument("body", help="Email body text. Instances of \{NAME\} and \{LAST\} will be replaced with first and last names.")
    parser.add_argument("-s", "--subject", help="Email subject line (optional).")

    args = parser.parse_args()
    
    pwd = os.getenv("GMAIL_APP_PWD")
    if not pwd:
        raise RuntimeError("Gmail password not found in \"GMAIL_APP_PWD\" environment variable.")
    # send_email("edwardyang125@gmail.com", "gheextnekztkmelw", "edward_yang_125@hotmail.com", "test", "test")
    parse_csv_and_send(
        args.sender,
        pwd,
        args.CSV,
        args.subject,
        args.body
    )
