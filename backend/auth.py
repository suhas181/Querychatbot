"""
auth.py — session handling and route protection

Replaces the browser-side auth.js from the frontend prototype. Credentials are
checked against the users table and the session cookie is signed by Flask, so
a student cannot reach an admin page by editing anything in the browser.
"""

from functools import wraps

from flask import flash, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from db import execute, query_one


def authenticate(email, password, role):
    """
    Return the user row on success, or an error message on failure.
    The two failure cases are kept separate so the login page can say which
    one it was, matching the behaviour the frontend already had.
    """
    email = (email or "").strip().lower()
    user = query_one(
        """SELECT u.*, d.name AS department
           FROM users u LEFT JOIN departments d ON d.id = u.department_id
           WHERE LOWER(u.email) = %s AND u.role = %s""",
        (email, role),
    )
    if not user:
        return None, (
            "No administrator account found with that email address."
            if role == "admin"
            else "No student account found with that email address."
        )
    if user["status"] != "Active":
        return None, "This account has been deactivated. Contact the college office."
    if not check_password_hash(user["password_hash"], password or ""):
        return None, "Incorrect password. Please try again."
    return user, None


def authenticate_student(email, password):
    """
    Sign in an existing student or create a student account on first login.

    This keeps student access simple for the college demo: a student can use
    any valid email address and choose a password. Existing student accounts
    still have their passwords checked normally. Administrator authentication
    remains unchanged and always uses the existing admin credentials.
    """
    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return None, "Email address and password are required."

    user = query_one(
        """SELECT u.*, d.name AS department
           FROM users u LEFT JOIN departments d ON d.id = u.department_id
           WHERE LOWER(u.email) = %s AND u.role = 'student'""",
        (email,),
    )

    if user:
        if user["status"] != "Active":
            return None, "This account has been deactivated. Contact the college office."
        if not check_password_hash(user["password_hash"], password):
            return None, "Incorrect password. Please try again."
        return user, None

    # Email is globally unique. Do not let a student accidentally take over
    # an existing administrator account.
    existing_role = query_one(
        "SELECT role FROM users WHERE LOWER(email) = %s",
        (email,),
    )
    if existing_role:
        return None, "This email address is already registered for an administrator account."

    # Use the email's local part as a friendly display name.
    local_part = email.split("@", 1)[0]
    name = " ".join(part.capitalize() for part in local_part.replace(".", " ").replace("_", " ").replace("-", " ").split()) or "Student"

    user_id = execute(
        """INSERT INTO users
             (role, name, email, password_hash, usn, department_id, semester)
           VALUES ('student', %s, %s, %s, NULL, NULL, NULL)""",
        (name[:80], email, generate_password_hash(password, method="pbkdf2:sha256")),
    )

    created_user = query_one(
        """SELECT u.*, d.name AS department
           FROM users u LEFT JOIN departments d ON d.id = u.department_id
           WHERE u.id = %s""",
        (user_id,),
    )
    return created_user, None


def login_user(user):
    session.clear()
    session["user_id"] = user["id"]
    session["role"] = user["role"]
    session.permanent = False


def logout_user():
    session.clear()


def current_user():
    """The signed-in user, re-read from the database on each request."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    return query_one(
        """SELECT u.*, d.name AS department
           FROM users u LEFT JOIN departments d ON d.id = u.department_id
           WHERE u.id = %s""",
        (user_id,),
    )


def login_required(role):
    """
    Protect a route. Anyone without a session of the right role is sent to the
    matching login page, so a student cannot open an admin URL directly.
    """
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if session.get("role") != role:
                flash("Please sign in to continue.", "error")
                return redirect(
                    url_for("admin_login" if role == "admin" else "student_login")
                )
            return view(*args, **kwargs)
        return wrapper
    return decorator


student_required = login_required("student")
admin_required = login_required("admin")
