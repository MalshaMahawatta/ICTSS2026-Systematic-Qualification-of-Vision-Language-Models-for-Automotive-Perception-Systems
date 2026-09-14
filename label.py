
import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, StringVar, BooleanVar
from PIL import Image, ImageTk
import shutil
from collections import Counter

class ImageLabelingApp:
    def __init__(self, root, image_dir="saved", labels_file="labels.json"):
        self.root = root
        self.root.title("Image Labeling Tool")
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.predefined_objects = [
            "car", "bus", "truck", "object", "bicycle", "motorcycle", 
            "pedestrian", "traffic light red", "traffic light green", 
            "traffic light yellow", "cone", "speed bump",""
        ]
        self.weather_options = ["clear", "rain", "fog"]
        self.image_files = []
        self.current_index = 0  # Index in the image_files list, not the image number
        self.labels_data = {}
        self.load_image_files()
        self.load_labels()
        main_frame = ttk.Frame(root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        left_pane = ttk.Frame(main_frame)
        left_pane.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
        
        right_pane = ttk.Frame(main_frame)
        right_pane.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
        self.image_label = ttk.Label(left_pane)
        self.image_label.grid(row=0, column=0, padx=2, pady=2)
        status_frame = ttk.Frame(left_pane)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        self.index_var = tk.StringVar()
        status_label1 = ttk.Label(status_frame, 
            text="Image: ", font=("Arial", 9))
        status_label1.grid(row=0, column=0, padx=2, sticky=tk.W)
        index_value = ttk.Label(status_frame, textvariable=self.index_var, font=("Arial", 9))
        index_value.grid(row=0, column=1, padx=2, sticky=tk.W)
        self.image_id_var = tk.StringVar()
        status_label2 = ttk.Label(status_frame, text="ID: ", font=("Arial", 9))
        status_label2.grid(row=1, column=0, padx=2, sticky=tk.W)
        image_id_value = ttk.Label(status_frame, textvariable=self.image_id_var, font=("Arial", 9))
        image_id_value.grid(row=1, column=1, padx=2, sticky=tk.W)
        self.labeled_var = tk.StringVar()
        status_label3 = ttk.Label(status_frame, text="Labeled: ", font=("Arial", 9))
        status_label3.grid(row=2, column=0, padx=2, sticky=tk.W)
        labeled_value = ttk.Label(status_frame, textvariable=self.labeled_var, font=("Arial", 9))
        labeled_value.grid(row=2, column=1, padx=2, sticky=tk.W)
        self.is_labeled_var = tk.StringVar()
        is_labeled_label = ttk.Label(status_frame, textvariable=self.is_labeled_var, font=("Arial", 9, "bold"))
        is_labeled_label.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky=tk.W)
        labeling_frame = ttk.LabelFrame(right_pane, text="Image Labels", padding=(3, 3, 3, 3))
        labeling_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=2, pady=2)
        settings_frame = ttk.Frame(labeling_frame)
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
        
        ttk.Label(settings_frame, text="Weather:", font=("Arial", 9)).grid(row=0, column=0, padx=2, sticky=tk.W)
        self.weather_var = StringVar()
        self.weather_var.set(self.weather_options[0])  # Default to clear
        weather_dropdown = ttk.Combobox(settings_frame, textvariable=self.weather_var, 
                                      values=self.weather_options, state="readonly", width=6, font=("Arial", 9))
        weather_dropdown.grid(row=0, column=1, padx=2, sticky=tk.W)
        weather_dropdown.bind("<<ComboboxSelected>>", lambda e: self.save_current_labels())
        self.dark_var = BooleanVar()
        self.dark_var.set(False)  # Default to not dark
        dark_checkbox = ttk.Checkbutton(settings_frame, text="Dark", variable=self.dark_var, 
                                        command=self.save_current_labels, style='Small.TCheckbutton')
        dark_checkbox.grid(row=0, column=2, padx=10, sticky=tk.W)
        style = ttk.Style()
        style.configure('Small.TCheckbutton', font=('Arial', 9))
        predefined_frame = ttk.LabelFrame(labeling_frame, text="Select Objects", padding=(3, 3, 3, 3))
        predefined_frame.grid(row=1, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        btn_row, btn_col = 0, 0
        for obj in self.predefined_objects:
            btn = ttk.Button(predefined_frame, text=obj, width=10, 
                            command=lambda o=obj: self.add_predefined_object(o))
            btn.grid(row=btn_row, column=btn_col, padx=1, pady=1)
            btn_col += 1
            if btn_col > 3:  # 4 buttons per row for smaller screen
                btn_col = 0
                btn_row += 1
        list_frame = ttk.Frame(labeling_frame)
        list_frame.grid(row=2, column=0, padx=2, pady=(5, 2), sticky=(tk.W, tk.E))
        
        ttk.Label(list_frame, text="Objects:", font=("Arial", 9)).grid(row=0, column=0, padx=2, sticky=tk.NW)
        
        list_container = ttk.Frame(list_frame)
        list_container.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        self.objects_listbox = tk.Listbox(list_container, height=4, width=25, font=("Arial", 9))
        self.objects_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        objects_scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.objects_listbox.yview)
        objects_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.objects_listbox.configure(yscrollcommand=objects_scrollbar.set)
        remove_button = ttk.Button(list_frame, text="Remove", width=8, 
                                  command=self.remove_object)
        remove_button.grid(row=0, column=2, padx=2, pady=2)
        counts_frame = ttk.LabelFrame(labeling_frame, text="Object Counts", padding=(3, 3, 3, 3))
        counts_frame.grid(row=3, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        counts_container = ttk.Frame(counts_frame)
        counts_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        counts_canvas = tk.Canvas(counts_container, highlightthickness=0, height=100)
        counts_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        counts_scrollbar = ttk.Scrollbar(counts_container, orient="vertical", command=counts_canvas.yview)
        counts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        counts_canvas.configure(yscrollcommand=counts_scrollbar.set)
        self.counts_frame = ttk.Frame(counts_canvas)
        counts_canvas.create_window((0, 0), window=self.counts_frame, anchor=tk.NW)
        self.counts_frame.bind("<Configure>", lambda e: counts_canvas.configure(scrollregion=counts_canvas.bbox("all")))
        nav_frame = ttk.Frame(labeling_frame)
        nav_frame.grid(row=4, column=0, padx=2, pady=(5, 2), sticky=(tk.W, tk.E))
        
        prev_btn = ttk.Button(nav_frame, text="◀ Prev", width=8, command=self.previous_image)
        prev_btn.grid(row=0, column=0, padx=2, pady=2)
        
        next_btn = ttk.Button(nav_frame, text="Next ▶", width=8, command=self.next_image)
        next_btn.grid(row=0, column=1, padx=2, pady=2)
        
        exit_btn = ttk.Button(nav_frame, text="Save & Exit", width=10, command=self.exit_app)
        exit_btn.grid(row=0, column=2, padx=2, pady=2)
        self.root.bind("<Left>", self.previous_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Delete>", self.remove_object)  # Delete key to remove selected object
        self.root.bind("<Escape>", self.exit_app)
        self.root.bind("<q>", self.exit_app)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        left_pane.columnconfigure(0, weight=1)
        left_pane.rowconfigure(0, weight=3)
        left_pane.rowconfigure(1, weight=1)
        
        right_pane.columnconfigure(0, weight=1)
        right_pane.rowconfigure(0, weight=1)
        self.find_last_labeled_image()
        if self.image_files:
            self.update_display()
        else:
            messagebox.showerror("Error", f"No images found in the '{self.image_dir}' directory!")
    
    def load_image_files(self):
        """Load all jpg files from the image directory"""
        self.image_files = []
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory '{self.image_dir}' not found")
            return
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith('.jpg'):
                self.image_files.append(filename)
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def load_labels(self):
        """Load existing labels from JSON file"""
        self.labels_data = {}
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    labels_list = json.load(f)
                    for item in labels_list:
                        image_id = item.get('id')
                        if image_id:
                            self.labels_data[image_id] = item
                print(f"Loaded {len(self.labels_data)} labeled images")
            except Exception as e:
                print(f"Error loading labels file: {e}")
                self.labels_data = {}
    
    def save_labels(self):
        """Save labels to JSON file"""
        try:
            labels_list = list(self.labels_data.values())
            with open(self.labels_file, 'w') as f:
                json.dump(labels_list, f, indent=2)
            print(f"Saved {len(labels_list)} labeled images")
            return True
        except Exception as e:
            print(f"Error saving labels: {e}")
            return False
    
    def find_last_labeled_image(self):
        """Find the last labeled image and set as current"""
        if not self.image_files:
            return
        for i, filename in enumerate(self.image_files):
            if filename in self.labels_data:
                self.current_index = i
                print(f"Starting with labeled image: {filename}")
                return
        self.current_index = 0
        self.update_labeled_count()
    
    def update_labeled_count(self):
        """Update the labeled images count"""
        labeled_count = 0
        for filename in self.image_files:
            if filename in self.labels_data:
                labeled_count += 1
        
        self.labeled_var.set(f"{labeled_count} / {len(self.image_files)}")
    
    def update_object_counts(self):
        """Update the object counts display"""
        for widget in self.counts_frame.winfo_children():
            widget.destroy()
        all_objects = list(self.objects_listbox.get(0, tk.END))
        object_counts = Counter(all_objects)
        row = 0
        for obj_type, count in sorted(object_counts.items()):
            obj_label = ttk.Label(self.counts_frame, text=obj_type, font=("Arial", 9))
            obj_label.grid(row=row, column=0, padx=2, pady=1, sticky=tk.W)
            count_label = ttk.Label(self.counts_frame, text=str(count), font=("Arial", 9))
            count_label.grid(row=row, column=1, padx=2, pady=1, sticky=tk.W)
            btn_frame = ttk.Frame(self.counts_frame)
            btn_frame.grid(row=row, column=2, padx=2, pady=1)
            increase_btn = ttk.Button(
                btn_frame, 
                text="▲", 
                width=1,
                command=lambda obj=obj_type: self.increase_object_count(obj)
            )
            increase_btn.pack(side=tk.LEFT, padx=1)
            decrease_btn = ttk.Button(
                btn_frame, 
                text="▼", 
                width=1,
                command=lambda obj=obj_type: self.decrease_object_count(obj)
            )
            decrease_btn.pack(side=tk.LEFT, padx=1)
            
            row += 1
    
    def increase_object_count(self, obj_type):
        """Increase the count of a specific object type"""
        self.objects_listbox.insert(tk.END, obj_type)
        self.update_object_counts()
        self.save_current_labels()  # Save immediately
    
    def decrease_object_count(self, obj_type):
        """Decrease the count of a specific object type"""
        all_items = list(self.objects_listbox.get(0, tk.END))
        for i in range(len(all_items) - 1, -1, -1):
            if all_items[i] == obj_type:
                self.objects_listbox.delete(i)
                break
                
        self.update_object_counts()
        self.save_current_labels()  # Save immediately
    
    def update_display(self):
        """Update the display with the current image and its labels"""
        if not self.image_files:
            return
        self.index_var.set(f"{self.current_index + 1} / {len(self.image_files)}")
        img_filename = self.image_files[self.current_index]
        self.image_id_var.set(img_filename)
        if img_filename in self.labels_data:
            self.is_labeled_var.set("LABELED ✓")
            image_data = self.labels_data[img_filename]
            self.weather_var.set(image_data.get('weather', 'clear'))
            self.dark_var.set(image_data.get('dark', False))
            self.objects_listbox.delete(0, tk.END)
            for obj in image_data.get('objects', []):
                self.objects_listbox.insert(tk.END, obj)
        else:
            self.is_labeled_var.set("NOT LABELED")
            self.objects_listbox.delete(0, tk.END)
        self.update_object_counts()
        self.update_labeled_count()
        try:
            img_path = os.path.join(self.image_dir, img_filename)
            image = Image.open(img_path)
            image.thumbnail((600, 475))
            if img_filename in self.labels_data:
                bordered = Image.new("RGB", (image.width + 6, image.height + 6), (0, 255, 0))
                bordered.paste(image, (3, 3))
                photo = ImageTk.PhotoImage(bordered)
            else:
                photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.configure(image='')
            self.image_label.image = None
            print(f"Error loading image {img_filename}: {e}")
    
    def add_predefined_object(self, obj_type):
        """Add a predefined object to the current image"""
        if not self.image_files:
            return
        self.objects_listbox.insert(tk.END, obj_type)
        self.update_object_counts()
        self.save_current_labels()
    
    def previous_image(self, event=None):
        """Navigate to the previous image"""
        if not self.image_files:
            return
        self.save_current_labels()
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self, event=None):
        """Navigate to the next image"""
        if not self.image_files:
            return
        self.save_current_labels()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_display()
    
    def save_current_labels(self):
        """Save labels for the current image and write to disk"""
        if not self.image_files:
            return
            
        img_filename = self.image_files[self.current_index]
        objects = list(self.objects_listbox.get(0, tk.END))
        if objects or self.weather_var.get() != "clear" or self.dark_var.get():
            self.labels_data[img_filename] = {
                'id': img_filename,
                'objects': objects,
                'weather': self.weather_var.get(),
                'dark': self.dark_var.get()
            }
        elif img_filename in self.labels_data:
            del self.labels_data[img_filename]
        self.save_labels()
        self.update_labeled_count()
        if img_filename in self.labels_data:
            self.is_labeled_var.set("LABELED ✓")
        else:
            self.is_labeled_var.set("NOT LABELED")
    
    def remove_object(self, event=None):
        """Remove selected object from listbox"""
        if not self.image_files:
            return
            
        try:
            selected_idx = self.objects_listbox.curselection()
            if selected_idx:
                self.objects_listbox.delete(selected_idx)
                self.update_object_counts()
                self.save_current_labels()
        except Exception:
            pass
    
    def exit_app(self, event=None):
        """Exit the application"""
        self.save_current_labels()
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

def main():
    root = tk.Tk()
    app = ImageLabelingApp(root)
    root.geometry("750x600")  # Smaller window size for 11-inch screen
    root.mainloop()

if __name__ == "__main__":
    main()