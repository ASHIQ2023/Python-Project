assignedtask = []

def display_tasks():
    if not assignedtask:
        print("No tasks found.")
    else:
        print("Tasks:")
        for i, task in enumerate(assignedtask, start=1):
            print(f"{i}. {task['title']} - Priority: {task['priority']}")

def add_task():
    title = input("Enter task title: ")
    priority = input("Enter task priority (High, Medium, Low): ").capitalize()
    assignedtask.append({"title": title, "priority": priority})
    print(f"Task '{title}' added!")

def delete_task():
    display_tasks()
    try:
        task_number = int(input("Enter the task number to delete: ")) - 1
        if 0 <= task_number < len(assignedtask):
            deleted_task = assignedtask.pop(task_number)
            print(f"Task '{deleted_task['title']}' deleted!")
        else:
            print("Invalid task number.")
            
    except ValueError:
        print("Invalid input. Please enter a valid task number.")


while True:
    print("To-Do List Program")
    print("1. Display tasks")
    print("2. Add a task")
    print("3. Delete a task")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":
        display_tasks()
    elif choice == "2":
        add_task()
    elif choice == "3":
        delete_task()
    elif choice == "4":
        break
    else:
        print("Invalid choice. Please try again.")

print("Exit the program")