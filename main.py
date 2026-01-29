# main.py
import tkinter as tk
from register_face import register_user
from recognize_face import recognize

def register():
    name = name_entry.get()
    if name:
        register_user(name)

def punch_in():
    recognize("Punch In")

def punch_out():
    recognize("Punch Out")

app = tk.Tk()
app.title("Face Attendance System")
app.geometry("300x300")

tk.Label(app, text="Enter Name").pack()
name_entry = tk.Entry(app)
name_entry.pack()

tk.Button(app, text="Register Face", command=register).pack(pady=10)
tk.Button(app, text="Punch In", command=punch_in).pack(pady=5)
tk.Button(app, text="Punch Out", command=punch_out).pack(pady=5)

app.mainloop()
