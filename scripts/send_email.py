#!/usr/bin/env python3

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes
from pathlib import Path


def _send(sender_email, pwd, recipient_email, msgstring):
    """
    Send an email via SMTP.

    This function connects to the Gmail SMTP server, logs in with the provided sender email and
    password, and sends an email to the recipient with the provided message string.

    Parameters:
    sender_email (str): Email address of the sender.
    pwd (str): Password or application-specific password for the sender's email account.
    recipient_email (str): Email address of the recipient.
    msgstring (str): The full email message as a string, including headers and body.

    Returns:
    None
    """

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(sender_email, pwd)
        server.sendmail(sender_email, recipient_email, msgstring)

    print("Successfully sent the email!")


def send_email(
    sender_name: str,
    sender_email: str,
    pwd: str,
    recipient_name: str,
    recipient_email: str,
    subject: str,
    body: str,
    format_dict: dict[str, str],
):
    """
    Send a formatted email.

    This function creates and sends an email using the provided sender and recipient information, subject, body content,
    and a dictionary for formatting the body content.

    Parameters:
    sender_name (str): Name of the email sender.
    sender_email (str): Email address of the sender.
    pwd (str): Password or application-specific password for the sender's email account.
    recipient_name (str): Name of the email recipient.
    recipient_email (str): Email address of the recipient.
    subject (str): Subject of the email.
    body (str): Body content of the email.
    format_dict (dict): Dictionary containing formatting information for the body content.

    Returns:
    None
    """

    FROM = f"{sender_name} <{sender_email}>"
    TO = f"{recipient_name} <{recipient_email}>"
    SUBJECT = subject
    TEXT = body.format_map(format_dict)

    message = "From: %s\nTo: %s\nSubject: %s\n\n%s".format_map(FROM, TO, SUBJECT, TEXT)

    _send(sender_email, pwd, recipient_email, message.as_string())


def send_email_with_embedded_image(
    sender_name: str,
    sender_email: str,
    pwd: str,
    recipient_name: str,
    recipient_email: str,
    subject: str,
    body: str,
    format_dict: str,
    imgpaths: list[Path],
):
    """
    Send an email with embedded images.

    This function creates and sends an email with embedded images using the provided sender and
    recipient information, subject, body content, and image file paths. The images are embedded
    within the email body using content IDs.

    Parameters:
    sender_name (str): Name of the email sender.
    sender_email (str): Email address of the sender.
    pwd (str): Password or application-specific password for the sender's email account.
    recipient_name (str): Name of the email recipient.
    recipient_email (str): Email address of the recipient.
    subject (str): Subject of the email.
    body (str): HTML body content of the email.
    format_dict (dict): Dictionary containing content IDs to be formatted into the email body.
    imgpaths (list[Path]): List of paths to the image files to be embedded in the email.

    Returns:
    None
    """

    msg = EmailMessage()

    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = f"{recipient_name} <{recipient_email}>"

    msg.set_content("")

    image_cids = []
    for i, ip in enumerate(imgpaths):
        image_cids.append(make_msgid())
        format_dict[f"image_cid{i}"] = image_cids[i][1:-1]

    msg.add_alternative(body.format_map(format_dict), subtype="html")

    # now open the image and attach it to the email
    for i, ip in enumerate(imgpaths):
        with open(ip, "rb") as img:
            maintype, subtype = mimetypes.guess_type(img.name)[0].split("/")
            msg.get_payload()[1].add_related(
                img.read(), maintype=maintype, subtype=subtype, cid=image_cids[i]
            )

    # Send the email
    _send(sender_email, pwd, recipient_email, msg.as_string())


if __name__ == "__main__":

    import argparse, os

    parser = argparse.ArgumentParser(
        prog="send_email.py",
        description="a simple CLI utility to send an email using gmail SMTP servers.",
    )

    parser.add_argument("sender_email", help="Email address to send email from.")
    parser.add_argument("sender_name", help="Name of the sender.")
    parser.add_argument("recipient_email", help="Email address to send email to.")
    parser.add_argument("recipient_name", help="Name of the recipient of the email.")
    parser.add_argument("body", help="Email body text.")
    parser.add_argument("-s", "--subject", help="Email subject line (optional).")

    args = parser.parse_args()

    pwd = os.getenv("GMAIL_APP_PWD")
    if not pwd:
        raise RuntimeError(
            'Gmail password not found in "GMAIL_APP_PWD" environment variable.'
        )
    # send_email("edwardyang125@gmail.com", "gheextnekztkmelw", "edward_yang_125@hotmail.com", "test", "test")
    send_email(
        args.sender_name,
        args.sender_email,
        pwd,
        args.recipient_name,
        args.recipient_email,
        args.subject,
        args.body,
    )
