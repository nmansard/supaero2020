def GepettoViewerServer(windowName="python-pinocchio", sceneName="world", loadModel=False):
    """
    Init gepetto-viewer by loading the gui and creating a window.
    """
    import gepetto.corbaserver
    try:
        viewer = gepetto.corbaserver.Client()
        gui = viewer.gui

        # Create window
        window_l = gui.getWindowList()
        if windowName not in window_l:
            gui.windowID = gui.createWindow(windowName)
        else:
            gui.windowID = gui.getWindowID(windowName)

        # Create scene if needed
        scene_l = gui.getSceneList()
        if sceneName not in scene_l:
            gui.createScene(sceneName)
            gui.addSceneToWindow(sceneName, gui.windowID)

        gui.sceneName = sceneName

        return gui

    except Exception:
        print("Error while starting the viewer client. ")
        print("Check whether gepetto-gui is properly started")
