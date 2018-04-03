module globals;
public import std.stdio, std.math, std.algorithm, std.traits, std.range,
              std.string, std.random;
public import gl3n.linalg;
alias float2 = vec2;
alias float3 = vec3;
alias float4 = vec4;

auto normalize(T)(T x){return x.normalized();}

/// I'm sure there's a DLang std func for this
/// truncates an element shared amongst an array
auto Truncate(string name, T)(inout T[] arr) {
  mixin(q{ typeof(__traits(getMember, T, "%s"))[] tarr; }.format(name));
  tarr.length = arr.length;
  foreach ( iter, a; arr )
    mixin(q{ tarr[iter] = a.%s; }.format(name));
  return tarr;
}

/// I'm also sure there's a DLang std func for this
/// reduces a range on an in-fix operator
auto OpReduce(string operation, Range)(Range range) {
  mixin(q{ return range.reduce!((x, y) => x %s y); }.format(operation));
}

/// uniform floa
float3 Sample_Float3 ( ) {
  return float3(uniform(-1.0f, 1.0f), uniform(-1.0f, 1.0f),
                uniform(-1.0f, 1.0f));
}

float Mix(float x, float y, float a) {
  return (x*(1.0f-a)) + y*a;
}
/// glsl Mix
float3 Mix(float3 x, float3 y, float a) {
  // can't divide with vectors in gl3n???? WTF?!?
  return float3(Mix(x.x, y.x, a),
                Mix(x.y, y.y, a),
                Mix(x.z, y.z, a));
}

import derelict.opengl, derelict.glfw3;
GLuint Load_Shaders(string vertex, string fragment) {
  GLuint vshader = glCreateShader(GL_VERTEX_SHADER),
         fshader = glCreateShader(GL_FRAGMENT_SHADER);

  void Check ( GLuint sh ) {
    GLint res;
    int info_log_length;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &res);
    glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &info_log_length);
    if ( info_log_length > 0 ){
      char[] msg; msg.length = info_log_length+1;
      glGetShaderInfoLog(sh, info_log_length, null, msg.ptr);
      writeln(msg);
      assert(false);
    }
  }

  immutable(char)* vertex_c   = toStringz(vertex),
                   fragment_c = toStringz(fragment);
  glShaderSource(vshader, 1, &vertex_c, null);
  glCompileShader(vshader);
  Check(vshader);

  glShaderSource(fshader, 1, &fragment_c, null);
  glCompileShader(fshader);
  Check(fshader);

  GLuint program_id = glCreateProgram();
  glAttachShader(program_id, vshader);
  glAttachShader(program_id, fshader);
  glLinkProgram(program_id);
  glDetachShader(program_id, vshader);
  glDetachShader(program_id, fshader);
  glDeleteShader(vshader);
  glDeleteShader(fshader);
  return program_id;
}
