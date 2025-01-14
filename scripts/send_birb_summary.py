#!/usr/bin/env python3

from send_email import send_email_with_embedded_image, send_email
import pandas as pd


def validate_df(df):

    has_right_columns = set(["NAME", "EMAIL"]).issubset(df.columns)

    if not has_right_columns:
        raise RuntimeError("Input CSV is missing NAME and EMAIL columns!")

    looks_like_email = df["EMAIL"].str.contains("@").all()

    if not looks_like_email:
        raise RuntimeError("At least one of the emails don't looke like an email!")


def parse_csv_and_send(
    sender_name, sender_email, pwd, csv_file, subject, body_template, imgpath=None
):

    df = pd.read_csv(csv_file)

    validate_df(df)

    df[["FIRST", "LAST"]] = df["NAME"].str.split(" ", n=1, expand=True)

    for _, row in df.iterrows():

        format_dict = {"FIRST": row["FIRST"], "LAST": row["LAST"], "NAME": row["NAME"]}
        if imgpath:
            send_email_with_embedded_image(
                sender_name,
                sender_email,
                pwd,
                row["NAME"],
                row["EMAIL"],
                subject,
                body_template,
                format_dict,
                imgpath,
            )
        else:
            send_email(
                sender_name,
                sender_email,
                pwd,
                row["NAME"],
                row["EMAIL"],
                subject,
                body_template,
                format_dict,
            )


if __name__ == "__main__":

    import argparse, os

    parser = argparse.ArgumentParser(
        prog="send_email_from_csv.py",
        description="a simple CLI utility to send an email using gmail SMTP servers, from a CSV file.",
    )

    parser.add_argument("sender_email", help="Email address to send email from.")
    parser.add_argument("sender_name", help="Name of sender.")
    parser.add_argument("CSV", help="CSV file to pull recipients from.")
    parser.add_argument(
        "body",
        help="Email body text. Instances of \{FIRST\} and \{LAST\} will be replaced with first and last names, respectively. \{NAME\} will be replaced with full name.",
    )
    parser.add_argument("-s", "--subject", help="Email subject line (optional).")
    parser.add_argument("-i", "--images", help="Path to image to send with the summary.", nargs="*")

    args = parser.parse_args()

    pwd = os.getenv("GMAIL_APP_PWD")
    if not pwd:
        raise RuntimeError(
            'Gmail password not found in "GMAIL_APP_PWD" environment variable.'
        )
    # send_email("edwardyang125@gmail.com", "gheextnekztkmelw", "edward_yang_125@hotmail.com", "test", "test")
    parse_csv_and_send(
        args.sender_name,
        args.sender_email,
        pwd,
        args.CSV,
        args.subject,
        args.body,
        args.images,
    )
