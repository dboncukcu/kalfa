def open_tui(root=".", config=None):
    from .app import KalfaApp

    KalfaApp(root=root, config=config).run()
