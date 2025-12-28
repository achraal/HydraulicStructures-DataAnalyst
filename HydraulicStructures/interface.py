import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from data_ouvrageHydraulique import (
    get_moyenne_ecart_type,
    afficher_matrice_centree_reduite,
    afficher_heatmap_correlation,
    get_inerties_table,
    plan_factoriel_individus,
    cercle_correlation,
    qualite_representation,
    contribution_individus_variables,
    afficher_clusters_k,
    pourcentage_clusters,
    metriques_random_forest,
    prediction_nouveaux_individus,
    afc_import_data_tableau,
    afc_matrice_frequences,
    afc_khi2_inertie_totale,
    afc_distances_khi2_cellules,
    afc_plan_factoriel,
    afc_test_khi2_interpretation,
    cyber_import_data,
    cyber_high_risk_summary,
    cyber_heatmap_isolation_only,
    cyber_heatmap_lof_only,
    cyber_if_lof_scatter,
)

P_START  = ("#22c55e", "#16a34a")  # vert bouton Start
P_PCA    = ("#0ea5e9", "#0284c7")  # bleu clair
P_AI     = ("#a855f7", "#7c3aed")  # violet
P_AFC    = ("#f97316", "#ea580c")  # orange
P_CYBER  = ("#ef4444", "#dc2626")  # rouge
P_GRAY   = ("#6b7280", "#4b5563")  # gris pour boutons retour

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    root = ctk.CTk()
    root.title("Hydraulic Structures Analyzer")
    root.geometry("1400x800")
    #main_frame = ttk.Frame(root, padding=10)
    main_frame = ctk.CTkFrame(root, fg_color="transparent")
    main_frame.pack(fill="both", expand=True)  # AJOUTEZ ÇA
    style = ttk.Style()
    style.theme_use("clam")
    # STYLES CustomTkinter PRO
    style.configure("Custom.Treeview",
        background="#1e1e1e", fieldbackground="#2d2d2d", 
        foreground="#e5e7eb", rowheight=28, font=("Segoe UI", 10))
    style.configure("Custom.Treeview.Heading",
        background="#3b82f6", foreground="white", font=("Segoe UI", 11, "bold"))
    
    # STYLES TABLEAUX PRO
    style.configure("Inertie.Treeview", background="#0f172a", fieldbackground="#1e293b", 
                    foreground="#f8fafc", rowheight=32, font=("Segoe UI", 11))
    style.configure("Inertie.Treeview.Heading", background="#10b981", foreground="white", 
                    font=("Segoe UI", 12, "bold"))
    
    style.configure("Contingence.Treeview", background="#0f172a", fieldbackground="#1e293b", 
                    foreground="#f8fafc", rowheight=32, font=("Consolas", 10))
    style.configure("Contingence.Treeview.Heading", background="#f59e0b", foreground="white", 
                    font=("Segoe UI", 12, "bold"))
    
    style.configure("Cyber.Treeview", background="#0f172a", fieldbackground="#1e293b", 
                    foreground="#f8fafc", rowheight=32, font=("Segoe UI", 11))
    style.configure("Cyber.Treeview.Heading", background="#ef4444", foreground="white", 
                    font=("Segoe UI", 12, "bold"))
    # STYLES RESTANTS
    style.configure("Stats.Treeview", background="#0f172a", fieldbackground="#1e293b", 
                    foreground="#f8fafc", rowheight=32, font=("Segoe UI", 11))
    style.configure("Stats.Treeview.Heading", background="#3b82f6", foreground="white", 
                    font=("Segoe UI", 12, "bold"))
    
    style.configure("Khi2.Treeview", background="#0f172a", fieldbackground="#1e293b", 
                    foreground="#f8fafc", rowheight=32, font=("Consolas", 10))
    style.configure("Khi2.Treeview.Heading", background="#8b5cf6", foreground="white", 
                    font=("Segoe UI", 12, "bold"))
    # Fix hover : TEXTE BLANC + FOND VERT FIXE
    style.map("Inertie.Treeview.Heading", 
              foreground=[("active", "white")],
              background=[("active", "#10b981")])  # Vert fixe
    
    style.map("Stats.Treeview.Heading", 
              foreground=[("active", "white")],
              background=[("active", "#3b82f6")])  # Bleu fixe
    
    style.map("Cyber.Treeview.Heading", 
              foreground=[("active", "white")],
              background=[("active", "#ef4444")])  # Rouge fixe
    
    style.map("Contingence.Treeview.Heading", 
              foreground=[("active", "white")],
              background=[("active", "#f59e0b")])  # Orange fixe
    
    style.map("Khi2.Treeview.Heading", 
              foreground=[("active", "white")],
              background=[("active", "#8b5cf6")])  # Violet fixe
    # ===== IA: K sélectionné =====
    screen_ai_kmenu = ctk.CTkFrame(main_frame, fg_color="#020617")
    # ===== Écran choix K (avant menu IA) =====
    title_ai_k = ctk.CTkLabel(
        screen_ai_kmenu,
        text="Choose K for K-Means",
        font=ctk.CTkFont("Segoe UI", 28, "bold"),
        text_color="#e5e7eb"
    )
    title_ai_k.pack(pady=30)
    
    kframe = ctk.CTkFrame(screen_ai_kmenu, fg_color="transparent")
    kframe.pack(fill="both", expand=True, padx=40, pady=20)
    
    kframe.columnconfigure((0, 1, 2), weight=1)
    kframe.rowconfigure(0, weight=1)
    
    ctk.CTkButton(
        kframe, text="K = 3", height=160, fg_color="#ec4899", hover_color="#be185d",
        text_color="white", corner_radius=12,
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        command=lambda: set_k_and_open_ai_menu(3)
    ).grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
    
    ctk.CTkButton(
        kframe, text="K = 4", height=160, fg_color="#ec4899", hover_color="#be185d",
        text_color="white", corner_radius=12,
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        command=lambda: set_k_and_open_ai_menu(4)
    ).grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
    
    ctk.CTkButton(
        kframe, text="K = 5", height=160, fg_color="#ec4899", hover_color="#be185d",
        text_color="white", corner_radius=12,
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        command=lambda: set_k_and_open_ai_menu(5)
    ).grid(row=0, column=2, padx=15, pady=15, sticky="nsew")

    def goto_ai_kmenu():
        show_screen(screen_ai_kmenu)
        
    selected_k = 4

    def set_k_and_open_ai_menu(k):
        nonlocal selected_k
        selected_k = k
        show_screen(screen_ai_menu)   # ton screenaimenu actuel
    # Navigation entre écrans
    def go_to_acp():
        show_screen(screen_acp_menu)
    
    def go_to_ai():
        show_screen(screen_ai_kmenu)
    
    def go_to_afc():
        show_screen(screen_afc_menu)
    
    def go_to_cyber():
        show_screen(screen_cyber_menu)
    
    def back_to_home():
        show_screen(screen_home)
        
    def go_to_home():
        show_screen(screen_home)
        
    def go_to_start():
        show_screen(screen_start)

    def back_to_acp():
        clear_content()
        show_screen(screen_acp_menu)
    
    def back_to_ai_home():
        clear_content()
        show_screen(screen_ai_menu)
    
    def back_to_afc_home():
        clear_content()
        show_screen(screen_afc_menu)
    
    def back_to_cyber_home():
        clear_content()
        show_screen(screen_cyber_menu)
    def back_to_afc():
        clear_content()
        show_screen(screen_afc_menu)

    def back_to_cyber():
        clear_content()
        show_screen(screen_cyber_menu)

    screen_start   = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_home    = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_acp_menu = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_result  = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_ai_menu = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_afc_menu = ctk.CTkFrame(main_frame, fg_color="#020617")
    screen_cyber_menu = ctk.CTkFrame(main_frame, fg_color="#020617")
    
    # ===== STYLES ÉCRAN RÉSULTAT DATA ANALYST =====
    style.configure(
        "Result.Title.TLabel",
        font=("Segoe UI", 28, "bold"),
        foreground="#e5e7eb",
        background="#020617"
    )
    style.configure(
        "Result.Back.TButton",
        font=("Segoe UI", 12, "bold"),
        padding=(20, 10),
        background="#3B82F6",  # bleu Data Analyst
        foreground="#FFFFFF",
        borderwidth=0
    )
    style.map(
        "Result.Back.TButton",
        background=[
            ("pressed", "#1D4ED8"),
            ("active", "#1D4ED8"),
            ("!active", "#3B82F6")
        ],
        foreground=[
            ("disabled", "#6b7280"),
            ("!disabled", "#FFFFFF")
        ]
    )
    # Style pour le cadre résultat (remplace LabelFrame)
    style.configure("Result.TFrame", background="#020617", borderwidth=0)

    for f in (screen_home, screen_acp_menu, screen_result,
              screen_ai_menu, screen_afc_menu, screen_cyber_menu, screen_ai_kmenu):
        f.pack_forget()
    
    def show_screen(frame):
        for f in (screen_start, screen_home, screen_acp_menu, screen_result,
                  screen_ai_menu, screen_afc_menu, screen_cyber_menu,screen_ai_kmenu):
            f.pack_forget()
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
    #selected_k = 4
    
    # ========== ÉCRAN START ==========
    screen_start = ctk.CTkFrame(main_frame, fg_color="#020617")
    start_container = ctk.CTkFrame(screen_start, fg_color="#020617", corner_radius=15)
    start_container.pack(fill="both", expand=True, padx=80, pady=80)
    
    main_title = ctk.CTkLabel(
        start_container,
        text="Data Analyst, AI and Cybersecurity Project\nin hydraulic structures",
        font=ctk.CTkFont("Segoe UI", 42, "bold"),
        text_color="#e5e7eb"
    )
    main_title.pack(pady=60)
    
    subtitle = ctk.CTkLabel(
        start_container,
        text="Data AI Analytics",
        font=ctk.CTkFont("Segoe UI", 16),
        text_color="#9ca3af"
    )
    subtitle.pack(pady=(10, 40))
    
    btn_change_k = ctk.CTkButton(
        screen_ai_menu,
        text="Change K",
        fg_color=P_GRAY[0],
        hover_color=P_GRAY[1],
        command=lambda: show_screen(screen_ai_kmenu)
    )
    btn_change_k.pack(pady=10)

    
    btn_start = ctk.CTkButton(
        start_container,
        text="🚀 Start",
        fg_color=P_START,
        hover_color=("#16a34a", "#15803d"),
        text_color=("white", "white"),
        font=ctk.CTkFont("Segoe UI", 26, "bold"),  # texte plus grand
        width=260,                                  # optionnel : largeur fixe
        height=140,                                 # plus haut qu'avant
        corner_radius=18,                           # bords plus arrondis
        command=go_to_home
    )
    btn_start.pack(pady=20)
    
    # --- Footer: Docteur à gauche / Developed by à droite ---
    footer_frame = ctk.CTkFrame(start_container, fg_color="transparent")
    footer_frame.pack(side="bottom", fill="x", pady=20, padx=20)
    
    footer_frame.grid_columnconfigure(0, weight=1)
    footer_frame.grid_columnconfigure(1, weight=1)
    
    lbl_docteur = ctk.CTkLabel(
        footer_frame,
        text="Dr ELMKHALET Mouna",
        font=ctk.CTkFont("Segoe UI", 18),
        text_color="#9ca3af"
    )
    lbl_docteur.grid(row=0, column=0, sticky="w")
    
    footer_label = ctk.CTkLabel(
        footer_frame,
        text="Developed by Achraf AIT LAHCEN v0.1 2025",
        font=ctk.CTkFont("Segoe UI", 18),
        text_color="#9ca3af"
    )
    footer_label.grid(row=0, column=1, sticky="e")

    # ========== ÉCRAN 1 : Accueil ==========
    style.configure(
        "Home.Title.TLabel",
        font=("Segoe UI", 24, "bold"),
        foreground="#e5e7eb",
        background="#020617"
    )
    
    style.configure(
        "Home.Subtitle.TLabel",
        font=("Segoe UI", 12),
        foreground="#9ca3af",
        background="#020617"
    )
    titre = ctk.CTkLabel(
        screen_home,
        text="Hydraulic Structure",
        font=ctk.CTkFont("Segoe UI", 24, "bold"),
        text_color="#e5e7eb"
    )
    titre.pack(pady=20)
    sous_titre = ctk.CTkLabel(
        screen_home,
        text="Click a button to start the analysis.",
        font=ctk.CTkFont("Segoe UI", 12),
        text_color="#9ca3af"
    )
    sous_titre.pack(pady=5)

    btns_home_frame = ctk.CTkFrame(screen_home, fg_color="#020617")
    btns_home_frame.pack(fill="both", expand=True, padx=20, pady=20)
    # après btns_home_frame = ttk.Frame(...)
    btns_home_frame.columnconfigure(0, minsize=350)
    btns_home_frame.rowconfigure(0, minsize=150)
    btns_home_frame.rowconfigure(1, minsize=150)
    btns_home_frame.columnconfigure(0, weight=1)
    btns_home_frame.columnconfigure(1, weight=1)
    btns_home_frame.rowconfigure(0, weight=1)
    btns_home_frame.rowconfigure(1, weight=1)
    btn_go_acp = ctk.CTkButton(
        btns_home_frame,
        text="📊 Data Analyst ACP",
        fg_color=P_PCA,
        hover_color=("#0284c7", "#0369a1"),
        text_color=("white", "white"),
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        height=70,
        corner_radius=12,
        command=go_to_acp
    )
    btn_go_acp.grid(row=0, column=0, padx=25, pady=20, sticky="nsew")
    
    # BOUTON 2 : AI
    btn_go_ai = ctk.CTkButton(
        btns_home_frame,
        text="🤖 Artificial Intelligence\nClustering & Forecasting",
        fg_color=P_AI,
        hover_color=("#7c3aed", "#5b21b6"),
        text_color=("white", "white"),
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        height=70,
        corner_radius=12,
        command=go_to_ai
    )
    btn_go_ai.grid(row=0, column=1, padx=25, pady=20, sticky="nsew")
    
    # BOUTON 3 : AFC
    btn_go_afc = ctk.CTkButton(
        btns_home_frame,
        text="📈 Data Analyst AFC",
        fg_color=P_AFC,
        hover_color=("#ea580c", "#c2410c"),
        text_color=("white", "white"),
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        height=70,
        corner_radius=12,
        command=go_to_afc
    )
    btn_go_afc.grid(row=1, column=0, padx=25, pady=20, sticky="nsew")
    
    # BOUTON 4 : CYBER
    btn_go_cyber = ctk.CTkButton(
        btns_home_frame,
        text="🛡️ Cyber Security",
        fg_color=P_CYBER,
        hover_color=("#dc2626", "#b91c1c"),
        text_color=("white", "white"),
        font=ctk.CTkFont("Segoe UI", 18, "bold"),
        height=70,
        corner_radius=12,
        command=go_to_cyber
    )
    btn_go_cyber.grid(row=1, column=1, padx=25, pady=20, sticky="nsew")

    # ========== ÉCRAN 3 : Résultat (CustomTkinter) ==========
    title_result = ctk.CTkLabel(screen_result, text="Results of the analysis", 
                               font=ctk.CTkFont("Segoe UI", 28, "bold"), 
                               text_color="#e5e7eb")
    title_result.pack(pady=30)
    
    top_buttons_result = ctk.CTkFrame(screen_result, fg_color="transparent")
    top_buttons_result.pack(pady=(0, 20))
    
    btn_back_to_acp = ctk.CTkButton(top_buttons_result, text="Return to PCA", fg_color="#3B82F6",
                                   font=ctk.CTkFont("Segoe UI", 12, "bold"), height=35,
                                   command=back_to_acp)
    btn_back_to_acp.pack(side="left", padx=10)
    
    btn_back_to_ai = ctk.CTkButton(top_buttons_result, text="Return to AI", fg_color="#10b981",
                                  font=ctk.CTkFont("Segoe UI", 12, "bold"), height=35,
                                  command=back_to_ai_home)
    btn_back_to_ai.pack(side="left", padx=10)
    btn_back_to_ai.pack_forget()
    
    btn_back_to_afc = ctk.CTkButton(top_buttons_result, text="Return to AFC", fg_color="#f59e0b",
                                   font=ctk.CTkFont("Segoe UI", 12, "bold"), height=35,
                                   command=back_to_afc)
    btn_back_to_afc.pack(side="left", padx=10)
    btn_back_to_afc.pack_forget()
    
    btn_back_to_cyber = ctk.CTkButton(top_buttons_result, text="Return to Cyber", fg_color="#ef4444",
                                     font=ctk.CTkFont("Segoe UI", 12, "bold"), height=35,
                                     command=back_to_cyber)
    btn_back_to_cyber.pack(side="left", padx=10)
    btn_back_to_cyber.pack_forget()
    
    def hide_all_back_buttons():
        btn_back_to_acp.pack_forget()
        btn_back_to_ai.pack_forget()
        btn_back_to_afc.pack_forget()
        btn_back_to_cyber.pack_forget()

    # ✅ SCROLL ANCIEN PRO
    result_scroll = ctk.CTkScrollableFrame(screen_result, fg_color="transparent", 
                                         corner_radius=15)
    result_scroll.pack(fill="both", expand=True, padx=20, pady=20)
    
    content_frame = ctk.CTkFrame(result_scroll, fg_color="#020617", corner_radius=12)
    #content_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    content_frame.pack(fill="x", expand=False, padx=20, pady=(10, 0))
    content = {"widget": None}  # figure OU tableau
    content["khi2_textbox"] = None

    # ===== Fonctions utilitaires =====
    def clear_content():
        if content["widget"] is not None:
            content["widget"].destroy()
            content["widget"] = None
            
        if content["khi2_textbox"] is not None:
            content["khi2_textbox"].destroy()
            content["khi2_textbox"] = None

    def show_table_inertie(columns, rows):
        clear_content()
        tv = ttk.Treeview(content_frame, columns=columns, show="headings", 
                         style="Inertie.Treeview",height=len(rows))
        #tv.pack(fill="both", expand=True, padx=15, pady=(15, 5))
        tv.pack(fill="x", expand=False, padx=15, pady=(10, 0))

        
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=160, stretch=True)
        
        tv.tag_configure("oddrow", background="#1e40af")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        content["widget"] = tv
    
    def show_table_contingence(columns, rows):
        clear_content()
        tv = ttk.Treeview(content_frame, columns=columns, show="headings", 
                         style="Contingence.Treeview",height=len(rows))
        #tv.pack(fill="both", expand=True, padx=15, pady=(15, 5))
        tv.pack(fill="x", expand=False, padx=15, pady=(15, 5))

        
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=120, stretch=True)
        
        tv.tag_configure("oddrow", background="#92400e")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        content["widget"] = tv
    
    def show_table_cyber(columns, rows):
        clear_content()
        tv = ttk.Treeview(content_frame, columns=columns, show="headings", 
                         style="Cyber.Treeview",height=len(rows))
        #tv.pack(fill="both", expand=True, padx=15, pady=(15, 5))
        tv.pack(fill="x", expand=False, padx=15, pady=(15, 5))
        
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=140, stretch=True)
        
        tv.tag_configure("oddrow", background="#dc2626")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        content["widget"] = tv
    
    def show_figure(fig):
        clear_content()
        canvas = FigureCanvasTkAgg(fig, master=content_frame)
        widget = canvas.get_tk_widget()
        #widget.pack(fill="both", expand=True, padx=10, pady=10)
        widget.pack(fill="x", expand=False, padx=20, pady=20)
        fig.set_tight_layout(True)
        canvas.draw()
        content["widget"] = widget
        
    def show_table_stats(columns, rows):  # Mean/Std + autres stats
        clear_content()
        tv = ttk.Treeview(content_frame, columns=columns, show="headings", 
                          style="Stats.Treeview",height=len(rows))
        #tv.pack(fill="both", expand=True, padx=15, pady=(15, 5))
        tv.pack(fill="x", expand=False, padx=15, pady=(15, 5))

        
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=160, stretch=True)
        
        tv.tag_configure("oddrow", background="#1e40af")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        content["widget"] = tv
    
    def show_table_khi2(columns, rows):  # χ² tableaux
        clear_content()
        tv = ttk.Treeview(content_frame, columns=columns, 
                          show="headings", style="Khi2.Treeview",height=len(rows))
        tv.pack(fill="x", expand=False, padx=15, pady=(15, 5))
       
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=120, stretch=True)
        
        tv.tag_configure("oddrow", background="#92400e")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        content["widget"] = tv
        
    # ====== Callbacks boutons ======

    def on_moyenne_ecart_type():
        df = get_moyenne_ecart_type()
        columns = ("Variable", "Mean", "Standard deviation")
        rows = [
            (row["Variable"], f"{row['Mean']:.2f}", f"{row['Standard deviation']:.2f}")
            for _, row in df.iterrows()
        ]
        show_screen(screen_result) 
        hide_all_back_buttons() # ← navigation
        btn_back_to_acp.pack(side="left", padx=10)  # montrer ACP
        # cacher IA
        show_table_stats(columns, rows)        # ← affichage

    def on_matrice_centree_reduite():
        fig = afficher_matrice_centree_reduite()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)  
        
        show_figure(fig)

    def on_heatmap_correlation():
        fig = afficher_heatmap_correlation()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)  
        
        show_figure(fig)

    def on_calcul_inerties():
        df = get_inerties_table()
        columns = ("Component", "Eigenvalues", "Inertia (%)", "Cumulative Inertia (%)")
        rows = [
            (
                row["Component"],
                f"{row['Eigenvalues']:.4f}",
                f"{row['Inertia (%)']:.2f}",
                f"{row['Cumulative Inertia (%)']:.2f}",
            )
            for _, row in df.iterrows()
        ]
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)   
        show_table_inertie(columns, rows)

    def on_plan_individus():
        fig = plan_factoriel_individus()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10) 
        show_figure(fig)

    def on_cercle_correlation():
        fig = cercle_correlation()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)  
        show_figure(fig)

    def on_qualite_representation():
        fig = qualite_representation()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)  
        show_figure(fig)

    def on_contribution():
        figs = contribution_individus_variables()
        fig_ind, fig_var = figs
        
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_acp.pack(side="left", padx=10)
        
        # DOUBLE FIGURE : utilise content_frame
        clear_content()
        container = ctk.CTkFrame(content_frame, fg_color="transparent")
        container.pack(fill="both", expand=True)
        
        # Gauche
        top_frame = ctk.CTkFrame(container, fg_color="#1e293b")
        top_frame.pack(fill="both", expand=True, pady=(0,10))  # ✅ HAUT
        ctk.CTkLabel(top_frame, text="Individuals", font=ctk.CTkFont("bold", 14)).pack(pady=5)
        canvas1 = FigureCanvasTkAgg(fig_ind, master=top_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill="both", expand=True)
        
        # Droite  
        bottom_frame = ctk.CTkFrame(container, fg_color="#1e293b")
        bottom_frame.pack(fill="both", expand=True)  # ✅ BAS (scroll auto)
        ctk.CTkLabel(bottom_frame, text="Variables", font=ctk.CTkFont("bold", 14)).pack(pady=5)
        canvas2 = FigureCanvasTkAgg(fig_var, master=bottom_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)
        
        content["widget"] = container

    def on_afficher_clusters():
        fig = afficher_clusters_k(selected_k)
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_ai.pack(side="left", padx=10)
        show_figure(fig)
    
    def on_pourcentage_clusters():
        fig = pourcentage_clusters(selected_k)
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_ai.pack(side="left", padx=10)
        show_figure(fig)
    
    def on_metriques_rf():
        df = metriques_random_forest(selected_k)
        columns = list(df.columns)
        rows = [
            [
                (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
                for v in row.values
            ]
            for _, row in df.iterrows()
        ]
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_ai.pack(side="left", padx=10)
        show_table_cyber(columns, rows)
    
    def on_prediction_nouveaux():
        fig = prediction_nouveaux_individus(selected_k)
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_ai.pack(side="left", padx=10)
        show_figure(fig)

    # ====== AFC CALLBACKS ======
    def on_import_afc():
        df = afc_import_data_tableau()
        columns = list(df.columns)
        rows = [list(row.values) for _, row in df.iterrows()]
        show_screen(screen_result) 
        hide_all_back_buttons() 
        btn_back_to_afc.pack(side="left", padx=10)           # montrer AFC
        show_table_contingence(columns, rows)

    def on_matrice_freq():
        fig = afc_matrice_frequences()
        show_screen(screen_result) 
        hide_all_back_buttons() 
        btn_back_to_afc.pack(side="left", padx=10)           
        show_figure(fig)

    def on_khi2_inertie():
        df = afc_khi2_inertie_totale()
    
        # --- texte long pour le textbox
        mask = df["Criteria"].astype(str).str.strip() == "Conclusion"
        conclusion_txt = df.loc[mask, "Values"].iloc[0] if mask.any() else "No Conclusion."
    
        # --- tableau avec texte court
        df_table = df.copy()
        if mask.any():
            df_table.loc[mask, "Values"] = "See explication below."
    
        columns = list(df_table.columns)
        rows = [[str(v) for v in row.values] for _, row in df_table.iterrows()]
    
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_afc.pack(side="left", padx=10)
    
        show_table_khi2(columns, rows)
    
        # --- textbox multi-ligne
        if content.get("khi2_textbox") is not None:
            content["khi2_textbox"].destroy()
            content["khi2_textbox"] = None
    
        tb = ctk.CTkTextbox(
            content_frame,
            height=170,
            wrap="word",
            font=ctk.CTkFont("Consolas", 12),
            fg_color="#0f172a",                 # fond
            text_color="#e5e7eb",               # texte
            border_color="#8b5cf6",             # bord (ex: violet)
            border_width=2,
            corner_radius=10,
            scrollbar_button_color="#334155",
            scrollbar_button_hover_color="#475569",
        )
        tb.pack(fill="x", padx=20, pady=(10, 15))
        tb.insert("0.0", conclusion_txt)   # ici tu peux avoir des \n [web:274]
        tb.configure(state="disabled")     # read-only [web:274]
        content["khi2_textbox"] = tb

    def on_khi2_distances():
        fig = afc_distances_khi2_cellules()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_afc.pack(side="left", padx=10)           
        show_figure(fig)

    def on_plan_afc():
        fig = afc_plan_factoriel()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_afc.pack(side="left", padx=10)           
        show_figure(fig)

    def on_test_khi2_interp():
        df = afc_test_khi2_interpretation()
    
        # --- texte long pour le textbox
        mask = df["Index"].astype(str).str.strip() == "Interpretation"
        interpretation_txt = df.loc[mask, "Value"].iloc[0] if mask.any() else "No Interpretation."
    
        # --- tableau avec texte court
        df_table = df.copy()
        if mask.any():
            df_table.loc[mask, "Value"] = "See explication below."
    
        columns = list(df_table.columns)
        rows = [[str(val) for val in row.values] for _, row in df_table.iterrows()]
    
        show_screen(screen_result)
        hide_all_back_buttons()
        btn_back_to_afc.pack(side="left", padx=10)
    
        show_table_khi2(columns, rows)
    
        # --- textbox multi-ligne
        if content.get("khi2_textbox") is not None:
            content["khi2_textbox"].destroy()
            content["khi2_textbox"] = None
    
        tb = ctk.CTkTextbox(
            content_frame,
            height=170,
            wrap="word",
            font=ctk.CTkFont("Consolas", 12),
            fg_color="#0f172a",                 # fond
            text_color="#e5e7eb",               # texte
            border_color="#8b5cf6",             # bord (ex: violet)
            border_width=2,
            corner_radius=10,
            scrollbar_button_color="#334155",
            scrollbar_button_hover_color="#475569",
        )
        tb.pack(fill="x", padx=20, pady=(10, 15))
        tb.insert("0.0", interpretation_txt)   # supporte \n [web:274]
        tb.configure(state="disabled")         # read-only [web:274]
    
        content["khi2_textbox"] = tb

    # ====== CYBER CALLBACKS ======
    def on_import_cyber():
        df = cyber_import_data()
        columns = list(df.columns)
        rows = [list(row.values) for _, row in df.iterrows()]
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_cyber.pack(side="left", padx=10)         
        show_table_cyber(columns, rows)

    def on_isolation_forest():
        fig = cyber_heatmap_isolation_only()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_cyber.pack(side="left", padx=10)          
        show_figure(fig)

    def on_lot_algorithm():
        fig = cyber_heatmap_lof_only()
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_cyber.pack(side="left", padx=10)           
        show_figure(fig)

    def on_high_risk_actions():
        df = cyber_high_risk_summary()
        columns = list(df.columns)
        rows = [[str(v) for v in row.values] for _, row in df.iterrows()]
        fig = cyber_if_lof_scatter()
        
        show_screen(screen_result)
        hide_all_back_buttons() 
        btn_back_to_cyber.pack(side="left", padx=10)
        
        clear_content()
        container = ctk.CTkFrame(content_frame, fg_color="transparent")
        container.pack(fill="x", expand=True, pady=10)
        
        # HAUT : Graph
        graph_frame = ctk.CTkFrame(container, fg_color="#1e293b", corner_radius=12)
        graph_frame.pack(fill="both", expand=True, pady=(0,15))
        ctk.CTkLabel(graph_frame, text="IF + LOF Scatter", 
                    font=ctk.CTkFont("Segoe UI", 16, "bold")).pack(pady=10)
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=(0,10))
        
        # BAS : Tableau High Risk
        table_frame = ctk.CTkFrame(container, fg_color="#1e293b", corner_radius=12)
        table_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(table_frame, text="High Risk Summary", 
                    font=ctk.CTkFont("Segoe UI", 16, "bold")).pack(pady=10)
        
        tv = ttk.Treeview(table_frame, columns=columns, show="headings", style="Cyber.Treeview",height=len(rows))
        tv.pack(fill="x", expand=False, padx=15, pady=(10, 0))
        
        for col in columns:
            tv.heading(col, text=col, anchor="center")
            tv.column(col, anchor="center", width=120, stretch=True)
        
        tv.tag_configure("oddrow", background="#dc2626")
        tv.tag_configure("evenrow", background="#1e293b")
        
        for idx, row in enumerate(rows):
            tag = "oddrow" if idx % 2 == 0 else "evenrow"
            tv.insert("", "end", values=row, tags=(tag,))
        
        content["widget"] = container
   
    # ========== ÉCRAN 2 : Menu ACP ==========
    title_acp = ctk.CTkLabel(screen_acp_menu, text="PCA Analyses", 
                            font=ctk.CTkFont("Segoe UI", 28, "bold"), 
                            text_color="#e5e7eb")
    title_acp.pack(pady=30)
    
    btn_back_home = ctk.CTkButton(screen_acp_menu, text="Return to Home",
                                 fg_color=P_GRAY,
    hover_color=("#4b5563", "#374151"),
    text_color=("white", "white"),
                                 font=ctk.CTkFont("Segoe UI", 14, "bold"),
                                 height=40, corner_radius=8, command=back_to_home)
    btn_back_home.pack(pady=20)
    
    # GRILLE 2x2 CENTRÉE
    acp_grid = ctk.CTkFrame(screen_acp_menu, fg_color="transparent")
    acp_grid.pack(fill="both", expand=True, padx=40, pady=20)
    
    acp_grid.columnconfigure(0, weight=1)
    acp_grid.columnconfigure(1, weight=1)
    acp_grid.rowconfigure(0, weight=1)
    acp_grid.rowconfigure(1, weight=1)
    acp_grid.rowconfigure(2, weight=1)
    acp_grid.rowconfigure(3, weight=1)
    acp_grid.rowconfigure(4, weight=1)
    
    ctk.CTkButton(acp_grid, text="📊 Mean and Standard Deviation Table", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_moyenne_ecart_type, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="🔢 Standardized Data Matrix", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_matrice_centree_reduite, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="🌡️Correlation Matrix", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_heatmap_correlation, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="⚖️ Calculation of Inertias", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_calcul_inerties, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="📈 Factorial Plane of Individuals", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_plan_individus, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=2, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="⭕ Correlation Circle in the Factorial Plane", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_cercle_correlation, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=2, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="⭐ Quality of Representation", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_qualite_representation, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=3, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(acp_grid, text="🎯 Contributions", height=80, fg_color=P_PCA, 
                  hover_color=("#0284c7", "#0369a1"), text_color=("white", "white"),
                  command=on_contribution, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=3, column=1, padx=15, pady=15, sticky="ew")

    # ========== ÉCRAN 5 : Menu AFC ==========
    title_afc = ctk.CTkLabel(screen_afc_menu, text="CA Analyses", 
                            font=ctk.CTkFont("Segoe UI", 28, "bold"), 
                            text_color="#e5e7eb")
    title_afc.pack(pady=30)
    
    btn_back_home_afc = ctk.CTkButton(screen_afc_menu, text="Return to Home",
                                     fg_color=P_GRAY,
    hover_color=("#4b5563", "#374151"),
    text_color=("white", "white"),
                                     font=ctk.CTkFont("Segoe UI", 14, "bold"),
                                     height=40, corner_radius=8, command=back_to_home)
    btn_back_home_afc.pack(pady=20)
    
    # GRILLE 2x2 XXL AFC
    afc_grid = ctk.CTkFrame(screen_afc_menu, fg_color="transparent")
    afc_grid.pack(fill="both", expand=True, padx=40, pady=20)
    afc_grid.columnconfigure(0, weight=1)
    afc_grid.columnconfigure(1, weight=1)
    afc_grid.rowconfigure(0, weight=1)
    afc_grid.rowconfigure(1, weight=1)
    
    ctk.CTkButton(afc_grid, text="📋 Import data Contingency table", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_import_afc, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(afc_grid, text="📊 Frequency Matrix", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_matrice_freq, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(afc_grid, text="Dependence χ2 total inertia", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_khi2_inertie, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(afc_grid, text="📏 χ2 distances (all calculations)", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_khi2_distances, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(afc_grid, text="📈 Factorial plane & associations", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_plan_afc, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=2, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(afc_grid, text="🔍 χ2 test interpretation", height=120, fg_color=P_AFC,
    hover_color=("#ea580c", "#c2410c"),
    text_color=("white", "white"),
                  command=on_test_khi2_interp, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=2, column=1, padx=15, pady=15, sticky="ew")
    
    # ========== ÉCRAN 6 : Menu Cyber Security (CustomTkinter) ==========
    title_cyber = ctk.CTkLabel(screen_cyber_menu, text="Cyber Analyses", 
                              font=ctk.CTkFont("Segoe UI", 28, "bold"), 
                              text_color="#e5e7eb")
    title_cyber.pack(pady=30)
    
    btn_back_home_cyber = ctk.CTkButton(screen_cyber_menu, text="Return to Home",
                                       fg_color=P_GRAY,
    hover_color=("#4b5563", "#374151"),
    text_color=("white", "white"),
                                       font=ctk.CTkFont("Segoe UI", 14, "bold"),
                                       height=40, corner_radius=8, command=back_to_home)
    btn_back_home_cyber.pack(pady=20)
    
    # GRILLE 2x2 XXL Cyber
    cyber_grid = ctk.CTkFrame(screen_cyber_menu, fg_color="transparent")
    cyber_grid.pack(fill="both", expand=True, padx=40, pady=20)
    cyber_grid.columnconfigure(0, weight=1)
    cyber_grid.columnconfigure(1, weight=1)
    cyber_grid.rowconfigure(0, weight=1)
    cyber_grid.rowconfigure(1, weight=1)
    
    ctk.CTkButton(cyber_grid, text="📊 Import Data", height=160, fg_color=P_CYBER,
    hover_color=("#dc2626", "#b91c1c"),
    text_color=("white", "white"),
                  command=on_import_cyber, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=0, padx=15, pady=15, sticky="ew")
    ctk.CTkButton(cyber_grid, text="🌲 Isolation Forest", height=160, fg_color=P_CYBER,
    hover_color=("#dc2626", "#b91c1c"),
    text_color=("white", "white"),
                  command=on_isolation_forest, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=1, padx=15, pady=15, sticky="ew")
    ctk.CTkButton(cyber_grid, text="🔍 LOF Algorithm", height=160, fg_color=P_CYBER,
    hover_color=("#dc2626", "#b91c1c"),
    text_color=("white", "white"),
                  command=on_lot_algorithm, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(cyber_grid, text="🚨 High-risk individuals and actions to take", height=160, fg_color=P_CYBER,
    hover_color=("#dc2626", "#b91c1c"),
    text_color=("white", "white"),
                  command=on_high_risk_actions, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=1, padx=15, pady=15, sticky="ew")
    
    # ========== ÉCRAN 4 : Menu IA (Clustering & Forecasting) ==========
    title_ai = ctk.CTkLabel(screen_ai_menu, text="AI Analyses", 
                           font=ctk.CTkFont("Segoe UI", 28, "bold"), 
                           text_color="#e5e7eb")
    title_ai.pack(pady=30)
    
    btn_back_home_ai = ctk.CTkButton(screen_ai_menu, text="Return to Home",
                                    fg_color=P_GRAY,
    hover_color=("#4b5563", "#374151"),
    text_color=("white", "white"),
                                    font=ctk.CTkFont("Segoe UI", 14, "bold"),
                                    height=40, corner_radius=8, command=back_to_home)
    btn_back_home_ai.pack(pady=20)
    btn_back_home_ai_k = ctk.CTkButton(
        screen_ai_kmenu,
        text="Return to Home",
        fg_color=P_GRAY,
        hover_color="#4b5563",
        text_color="white",
        font=ctk.CTkFont("Segoe UI", 14, "bold"),
        height=40,
        corner_radius=8,
        command=back_to_home
    )
    btn_back_home_ai_k.pack(pady=20)
    
    # GRILLE 2x2 XXL AI
    ai_grid = ctk.CTkFrame(screen_ai_menu, fg_color="transparent")
    ai_grid.pack(fill="both", expand=True, padx=40, pady=20)
    
    ai_grid.columnconfigure(0, weight=1)
    ai_grid.columnconfigure(1, weight=1)
    ai_grid.rowconfigure(0, weight=1)
    ai_grid.rowconfigure(1, weight=1)
    
    ctk.CTkButton(ai_grid, text="🎨 Display of the clusters", height=160, fg_color=P_AI,
              hover_color=("#7c3aed", "#5b21b6"),
              text_color=("white", "white"),
                  command=on_afficher_clusters, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(ai_grid, text="📊 Percentage of each cluster in the data", height=160, fg_color=P_AI,
              hover_color=("#7c3aed", "#5b21b6"),
              text_color=("white", "white"),
                  command=on_pourcentage_clusters, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=0, column=1, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(ai_grid, text="🌲 Random Forest training metrics", height=160, fg_color=P_AI,
              hover_color=("#7c3aed", "#5b21b6"),
              text_color=("white", "white"),
                  command=on_metriques_rf, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=0, padx=15, pady=15, sticky="ew")
    
    ctk.CTkButton(ai_grid, text="🔮 Prediction for new individuals", height=160, fg_color=P_AI,
              hover_color=("#7c3aed", "#5b21b6"),
              text_color=("white", "white"),
                  command=on_prediction_nouveaux, corner_radius=12,
                  font=ctk.CTkFont("Segoe UI", 16, "bold")).grid(row=1, column=1, padx=15, pady=15, sticky="ew") 
    # Écran initial
    show_screen(screen_start)
    root.mainloop()

if __name__ == "__main__":
    main()