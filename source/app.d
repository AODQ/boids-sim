import derelict.opengl;
import derelict.glfw3;
import globals;
static import boidsim;
static import render;

GLFWwindow* window;
int window_width = 320, window_height = 240;
// int window_width = 640, window_height = 480;
// int window_width = 1440, window_height = 1080;

void main ( ) {
  DerelictGL3.load();
  DerelictGLFW3.load();
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
  glfwWindowHint(GLFW_RESIZABLE,      GL_FALSE                 );
  glfwWindowHint(GLFW_FLOATING,       GL_TRUE                  );
  glfwWindowHint(GLFW_REFRESH_RATE,  0                         );
  glfwSwapInterval(1);

  window = glfwCreateWindow(window_width, window_height, "boids", null, null);

  glfwWindowHint(GLFW_FLOATING, GL_TRUE);
  glfwMakeContextCurrent(window);
  DerelictGL3.reload();
  glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);

  glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDisable(GL_DEPTH_TEST);

  boidsim.Initialize();
  render.Initialize();

  float time=0.0f;
  while ( !glfwWindowShouldClose(window) &&
           glfwGetKey(window, GLFW_KEY_Q) != GLFW_PRESS &&
           glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS ) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float frame_time = glfwGetTime();
    float delta = (frame_time-time);
    time = frame_time;

    boidsim.Boid_Iterate(time, delta);
    render.Render(time);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}
