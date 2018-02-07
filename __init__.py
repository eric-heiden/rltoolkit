from . import *

# XXX This is a bug where dm_control cannot find a GL backend
try:
    from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
except:
    pass

try:
    from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
except:
    pass

try:
    from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
except:
    pass
