#!/usr/bin/env python3

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import mimetypes

def _send(sender_email, pwd, recipient_email, msgstring):

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(sender_email, pwd)
        server.sendmail(sender_email, recipient_email, msgstring)

    print("Successfully sent the email!")

def send_email(sender_name, sender_email, pwd, recipient_name, recipient_email, subject, body, format_dict):

    FROM = f"{sender_name} <{sender_email}>"
    TO = f"{recipient_name} <{recipient_email}>"
    SUBJECT = subject
    TEXT = body.format_map(format_dict)

    message = "From: %s\nTo: %s\nSubject: %s\n\n%s".format_map(FROM, TO, SUBJECT, TEXT)

    _send(sender_email, pwd, recipient_email, message.as_string())

def send_email_with_embedded_image(sender_name, sender_email, pwd, recipient_name, recipient_email, subject, body, format_dict, imgpath):

    msg = EmailMessage()

    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = f"{recipient_name} <{recipient_email}>"

    msg.set_content("")

    image_cid = make_msgid()

    format_dict["image_cid"] = image_cid[1:-1]

    msg.add_alternative(body.format_map(format_dict), subtype='html')

    # now open the image and attach it to the email
    with open(imgpath, 'rb') as img:

        # know the Content-Type of the image
        maintype, subtype = mimetypes.guess_type(img.name)[0].split('/')

        # attach it
        msg.get_payload()[1].add_related(img.read(), 
                                            maintype=maintype, 
                                            subtype=subtype, 
                                            cid=image_cid)
        
    
    # Send the email
    _send(sender_email, pwd, recipient_email, msg.as_string())

if __name__=="__main__":

    import argparse, os

    parser = argparse.ArgumentParser(
        prog = "send_email.py",
        description = "a simple CLI utility to send an email using gmail SMTP servers.")
    
    parser.add_argument("sender_email", help="Email address to send email from.")
    parser.add_argument("sender_name", help="Name of the sender.")
    parser.add_argument("recipient_email", help="Email address to send email to.")
    parser.add_argument("recipient_name", help="Name of the recipient of the email.")
    parser.add_argument("body", help="Email body text.")
    parser.add_argument("-s", "--subject", help="Email subject line (optional).")

    args = parser.parse_args()
    
    pwd = os.getenv("GMAIL_APP_PWD")
    if not pwd:
        raise RuntimeError("Gmail password not found in \"GMAIL_APP_PWD\" environment variable.")
    # send_email("edwardyang125@gmail.com", "gheextnekztkmelw", "edward_yang_125@hotmail.com", "test", "test")
    send_email(
        args.sender_name,
        args.sender_email,
        pwd,
        args.recipient_name,
        args.recipient_email,
        args.subject,
        args.body
    )
