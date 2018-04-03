module render;
import derelict.opengl, derelict.glfw3;
import globals;
static import boidsim;
import imageformats : read_png;

GLuint vao, screen_vbo, program_id;
GLuint boid_ori_uloc, boid_dir_uloc, boid_ori_mass_uloc;
GLuint[] textures, textures_uloc;
GLuint time_uloc, cam_range_uloc;
immutable GLfloat[] screen_vbo_data = [
  -1.0f, -1.0f, 0.0f,
  1.0f, -1.0f, 0.0f,
  -1.0f, 1.0f, 0.0f,

  1.0f, 1.0f, 0.0f,
  -1.0f, 1.0f, 0.0f,
  1.0f, -1.0f, 0.0f,
];


void Initialize () {
  // load program
  program_id = Load_Shaders(vertex, fragment);
  // load buffers
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &screen_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, screen_vbo);
  glBufferData(GL_ARRAY_BUFFER, GLfloat.sizeof*screen_vbo_data.length,
               screen_vbo_data.ptr, GL_STATIC_DRAW);
  // load uniforms
  time_uloc = glGetUniformLocation(program_id, "u_time".ptr);
  cam_range_uloc = glGetUniformLocation(program_id, "u_cam_range".ptr);
  boid_ori_uloc = glGetUniformLocation(program_id, "boid_ori");
  boid_dir_uloc = glGetUniformLocation(program_id, "boid_dir");
  boid_ori_mass_uloc = glGetUniformLocation(program_id, "boid_ori_mass");
  textures_uloc = [
    glGetUniformLocation(program_id, "u_texture_0"),
    glGetUniformLocation(program_id, "u_texture_1"),
    glGetUniformLocation(program_id, "u_texture_2"),
  ];
  // set uniform textures
  glActiveTexture(GL_TEXTURE0); glUniform1f(0, 0);
  glActiveTexture(GL_TEXTURE1); glUniform1f(1, 1);
  glActiveTexture(GL_TEXTURE2); glUniform1f(2, 2);
  // load textures;
  textures.length = 3;
  glGenTextures(3, textures.ptr);

  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glActiveTexture(0);
  auto texture_0 = read_png("textures/texture_0.png");
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA,
                              GL_UNSIGNED_BYTE, texture_0.pixels.ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  auto texture_1 = read_png("textures/texture_1.png");
  glActiveTexture(1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA,
                              GL_UNSIGNED_BYTE, texture_1.pixels.ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  auto texture_2 = read_png("textures/texture_2.png");
  glActiveTexture(2);
  glBindTexture(GL_TEXTURE_2D, textures[2]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 1024, 0, GL_RGBA,
                              GL_UNSIGNED_BYTE, texture_2.pixels.ptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

string vertex = q{#version 330 core
  layout(location = 0) in vec3 vertex_pos;
  out vec2 frag_coord;
  void main ( ) {
    gl_Position = vec4(vertex_pos, 1.0f);
    frag_coord.x = vertex_pos.x;
    frag_coord.y = vertex_pos.y;
  }
};

void Render ( float glfw_time ) {
  // update boids uniform
  GLfloat[] boid_ori_data; boid_ori_data.length = 3*boidsim.Boid_amt;
  GLfloat[] boid_dir_data; boid_dir_data.length = 3*boidsim.Boid_amt;
  GLfloat[] boid_ori_mass_data; boid_ori_mass_data.length = 3;
  foreach ( iter, ref boid; boidsim.boids ) {
    boid_ori_data[iter*3 .. iter*3+3] = boid.ori.vector;
    boid_dir_data[iter*3 .. iter*3+3] = boid.dir.vector;
  }
  boid_ori_mass_data[] = boidsim.ori_mass.vector;
  // render and stuff
  glUseProgram(program_id);
  // screen vbo
  glBindVertexArray(vao);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, screen_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, null);
  // uniforms
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glUniform1i(0, 0);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glUniform1i(1, 1);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, textures[2]);
  glUniform1i(2, 2);
  glUniform1f(time_uloc, glfw_time);
  glUniform1f(cam_range_uloc, boidsim.cam_range);
  glUniform3fv(boid_ori_uloc, 3*boidsim.Boid_amt, boid_ori_data.ptr);
  glUniform3fv(boid_dir_uloc, 3*boidsim.Boid_amt, boid_dir_data.ptr);
  glUniform3fv(boid_ori_mass_uloc, 1, boid_ori_mass_data.ptr);
  // render
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glDisableVertexAttribArray(0);
}

//-----------------------------------------------------------------------------
//---- frag -------------------------------------------------------------------
//-----------------------------------------------------------------------------
string fragment = q{#version 330 core
#extension GL_ARB_explicit_uniform_location : enable
#define PI   3.141592653589793
#define IPI  0.318309886183791
#define IPI2 0.159154943091895
#define TAU  6.283185307179586
#define ITAU 0.159154943091895
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define BOID_AMT %s

in vec2 frag_coord;
out vec4 frag_colour;

layout(location=0) uniform sampler2D u_texture_0;
layout(location=1) uniform sampler2D u_texture_1;
layout(location=2) uniform sampler2D u_texture_2;
uniform float3 boid_ori[BOID_AMT];
uniform float3 boid_dir[BOID_AMT];
uniform float3 boid_ori_mass;
uniform float u_time;
uniform float u_cam_range;

float saturate ( float t ) { return clamp(t, 0.0f, 1.0f); }
float3 saturate ( float3 t ) {return clamp(t, float3(0.0f), float3(1.0f));}

struct Ray {
  float3 ori, dir;
};

mat3 Look_At ( float3 origin, float3 center, float3 up ) {
  float3 ww = normalize(normalize(center-origin)),
          uu = normalize(cross(up, ww)),
          vv = normalize(cross(ww, uu));
  return mat3(vv, uu, ww);
}
Ray Look_At ( float2 uv, float3 origin, float3 center, float3 up, float fov){
  mat3 la = Look_At(origin, center, up);
  return Ray(origin, normalize(la*float3(uv.y, uv.x, fov)));
}

float2 Map(float3 o);
float2 March ( Ray ray ) {
  float dist = 0.0f;
  float2 cur;
  for ( int i = 0; i != 256; ++ i ) {
    cur = Map(ray.ori + ray.dir*dist);
    if ( cur.x <= 0.001f || dist > 128.0f ) break;
    dist += cur.x;
  }
  if ( dist > 128.0f ) return float2(-1.0f);
  return float2(dist, cur.y);
}

float AO_March ( float3 ro, float3 rd ) {
  ro += rd*0.10f;
  float dist = 0.0f;
  float cur;
  for ( int i = 0; i != 16; ++ i ) {
    cur = Map(ro + rd*dist).x;
    if ( cur <= 0.01f || dist > 4.0f ) break;
    dist += cur;
  }
  if ( dist > 4.0f ) return 1.0f;
  return dist/16.0f;
}

/// just a cheap guess at AO
float Ambient_Occlusion ( float3 o ) {
  float2 e = float2(1.0f, -1.0f);
  return (
    AO_March(o, float3(0.0f, 1.0f, 0.0f)) +
    AO_March(o, normalize(float3(0.1f, -1.0f, 0.2f)))
  );
}

float Displacement(float3);
float3 Water_Normal ( float3 o, float t ) {
  // we know the normal is up
  float3 N = float3(0.0f, 1.0f, 0.0f);
  N.x += Displacement(o*1.2f);
  N.z += Displacement(o*1.5f);
  return normalize(N);
}

float3 Normal ( float3 o, float t ) {
  float2 e = float2(1.0f, -1.0f)*0.005f*t;
  return normalize(
    e.xyy*Map(o + e.xyy).x +
    e.yyx*Map(o + e.yyx).x +
    e.yxy*Map(o + e.yxy).x +
    e.xxx*Map(o + e.xxx).x
  );
}

/// heightmap more or less from IQ's Canyon
float4 texture_good(sampler2D sampler, float2 uv, float lo ) {
  uv = uv*1024.0f - 0.5f;
  float2 iuv = floor(uv);
  float2 f = fract(uv);

  float4 rg1 = textureLod( sampler, (iuv + float2(0.5f,0.5f))/1024.0f, lo);
  float4 rg2 = textureLod( sampler, (iuv + float2(1.5f,0.5f))/1024.0f, lo);
  float4 rg3 = textureLod( sampler, (iuv + float2(0.5f,1.5f))/1024.0f, lo);
  float4 rg4 = textureLod( sampler, (iuv + float2(1.5f,1.5f))/1024.0f, lo);
  return mix( mix(rg1,rg2,f.x), mix(rg3,rg4,f.x), f.y );
}

float noise1 ( float3 x ) {
  float3 p = floor(x);
  float3 f = fract(x);
  // quintic
  float3 df = 30.0*f*f*(f*(f-2.0)+1.0);
  f = f*f*f*(f*(f*6.-15.)+10.);
  float2 uv = (p.xy + float2(37.0f, 17.0f)*p.z) + f.x;
  float2 rg = textureLod(u_texture_2, (uv+1.1f)/1024.0f, 0.0f).yx+df.xz/10.0f;
  return mix(rg.x, rg.y, f.z);
}

const mat3 m = mat3(0.0f, 0.8f, 0.6f,
                    -0.8f, 0.36f, -0.48f,
                    -0.6f, 0.48f, 0.64f);
float Displacement ( float3 p ) {
  float f = 0.0f;
  f += 0.5000f*noise1(p); p = m*p*2.02f;
  f += 0.2500f*noise1(p); p = m*p*2.03f;
  // f += 0.1250f*noise1(p); p = m*p*2.01f;
  // f += 0.0625f*noise1(p);
  return f;
}

float Noise1 ( sampler2D sampler, float2 uv ) {
  return sin(uv.x*1.3f) + sin(uv.y*1.6f);
}

float FBM ( sampler2D sampler, in float2 uv ) {
  const mat2 m = mat2(0.8, 0.6, -0.6, 0.8);
  float f = 0.0;
  f += 0.500000*(0.5 + 0.5*Noise1(sampler, uv)); uv = m*uv*2.02;
  f += 0.250000*(0.5 + 0.5*Noise1(sampler, uv)); uv = m*uv*2.03;
  f += 0.125000*(0.5 + 0.5*Noise1(sampler, uv)); uv = m*uv*2.01;
  f += 0.062500*(0.5 + 0.5*Noise1(sampler, uv)); uv = m*uv*2.04;
  f += 0.031250*(0.5 + 0.5*Noise1(sampler, uv));
  return f/0.9375;
}

float Pattern ( sampler2D sampler, in float2 uv, out float2 q, out float2 r ){
  q = float2(FBM(sampler, uv + float2(0.0, 0.2f)),
              FBM(sampler, uv + float2(4.2, 2.1f)));
  r = float2(FBM(sampler, uv + 4.0*q + float2(1.6, 0.2)),
              FBM(sampler, uv + 4.0*q + float2(0.3, 1.3f)));
  return FBM(sampler, uv+q-r*0.25);
}

float Terrain ( in float2 p ) {
  float th = smoothstep(0.0f, 1.0f, texture(u_texture_2, 0.001f*p, 0.0f).x);
  float rr = smoothstep(0.1f, 0.5f, texture(u_texture_1, 0.003f*p, 0.0f).y);
  float h = 0.0f;
  h -= (4.5f*rr);
  h += (th*7.0f);
  return -h;
}

void opRotate(inout float2 p, float a) {
  p = cos(a)*p + sin(a)*float2(p.y, -p.x);
}

float sdSphere ( float3 o, float radius ) { return length(o)-radius; }
float sdTorus ( float3 o, float2 t ) {
  float2 q = float2(length(o.xz) - t.x, o.y);
  return length(q) - t.y;
}

float Boid_Model ( float3 ori, float3 dir ) {
  // mat3 la = Look_At(ori, ori+dir, float3(0.0f, 1.0f, 0.0f));
  // ori = inverse(la)*ori;
  float sph = max(-sdSphere(ori+dir*0.1f, 0.05f),
                    sdSphere(ori, 0.1f));
  opRotate(ori.xy, dir.x*PI);
  sph = min(sph, sdTorus(ori, float2(0.2f, 0.01f)));
  return sph;
}

void Union(inout float2 t, float d, float ID ) {
  if ( t.x > d ) t = float2(d, ID);
}

float sdDisk(float3 p, float r) {
  float l = length(p.xz) - r;
  return l < 0.0f ? abs(p.y) : length(float2(p.y, l));
}

float sdLine(float3 p, float3 a, float3 b) {
  float3 ab = b - a;
  float t = clamp(dot(p - a, ab)/dot(ab, ab), 0.0f, 1.0f);
  return length((ab*t + a) - p);
}

float sdMushroom ( float3 O, float cap, float height, float2 twist ) {
  opRotate(O.xz, twist.x);
  float c = cos(twist.y*0.2f*O.y);
  float s = sin(twist.y*0.2f*O.y);
  mat2 m = mat2(c, -s, s, c);
  O = float3(m*O.xy, O.z);
  float t = sdSphere(O, 0.8f);
  t = max(-sdSphere(O-float3(0.0f, -0.7f+mix(0.0f, 0.7f, cap), 0.0f), 0.9f), t);
  t = min(sdLine(O, float3(0.0f, 0.6f, 0.0f), float3(0.0f, -height, 0.0f))-0.1f, t);
  t += texture(u_texture_0, O.xz).x*0.04f;
  return t;
}

float sdBox ( float3 o, float3 b ) {
  float3 d = abs(o) - b;
  return min(max(d.x, max(d.y, d.z)), 0.0f) + length(max(d, float3(0.0f)));
}

float2 Map ( float3 o ) {
  // heightmap
  float height = Terrain(o.xz*0.15f);
  float dis = Displacement(0.15f*o*float3(1.0f, 4.0f, 1.2f))*3.0f;
  float2 dmin = float2(999.0f);
  dmin = float2((dis + o.y-height)*0.25f, 1.0f);
  { // shitty idk
    float3 q = o;
    float2 size = float2(100.0f, 30.0f);
    float2 hs = size*0.5f;
    float2 id = floor((q.xz+hs)/size);
    float2 bid = floor((q.xz+hs))/30.0f;
    q.y += sin(o.z*0.5f)*1.2f;
    q.xz = mod(q.xz+hs+u_time*2.0f, size) - hs;
    opRotate(q.xy, fract(sin(sin(bid.x*232.0232f)+bid.y)*239.232f)+
                   sin(u_time*fract(bid.y))*0.2f);
    float dist = sdBox(q, float3(3.0f, 1.0f, 24.0f));
    Union(dmin, dist*0.5f, 4.0f);
    q = o;
    q.y -= height;
    hs = float2(150.0f*0.5f);
    id = floor((q.xz+hs)/150.0f);
    q.xz = mod(q.xz+hs, 150.0f) - hs;
    opRotate(q.xy, fract(sin(id.x*358.23f)*2392.0f));
    opRotate(q.xz, fract(sin(id.x*358.23f)*2392.0f));
    dist = sdBox(q, float3(8.0f, 8.0f, 8.0f));
    Union(dmin, dist*0.5f, 4.0f);
  }
  // boids
  // for ( int i = 1; i != BOID_AMT; ++ i ) {
  //   float3 p = boid_ori[i]+o;
  //   Union(dmin, Boid_Model(boid_ori[i]+o, boid_dir[i]), 2.0f);
  // }
  // water
  float water_disp = sin(o.z*0.1f+u_time*0.2f)*0.4f - cos(o.x*0.2f+u_time)*0.1f;
  Union(dmin, (abs(o.y+4.1f-water_disp)-0.0001f*0.5f)*0.5f, 3.0f);
  return dmin;
}

const float3 Sun_dir = normalize(float3(-0.2f, -0.05f, -0.2f));
const float3 Sun_col = float3(1.0f, 1.0f, 0.868f);
const float3 Sun_dif = pow(Sun_col*40.0f, float3(1.2f, 1.0f, 1.4f));

float3 Cubemap(float3 dir) {
  float3 ambient = float3(0.0f);
  ambient = mix(float3(0.03f, 0.03f, 0.12f), float3(0.18f, 0.03f, 0.12f),
                float3(dir.y));
  float sun = dot(dir, float3(-Sun_dir.x, Sun_dir.y, -Sun_dir.z));
  if ( sun <= -0.99f ) {
    ambient = Sun_col;
  }
  return ambient;
}

float3 Shade_Boid ( float3 O, float3 N, float3 wi, float3 sha,
                            float2 info ) {
  float3 wo = normalize(Sun_dir);
  return float3(0.1f) + float3(clamp(dot(N, wo), 0.0f, 1.0f));
}

float GTerm ( float3 N, float3 V, float k ) {
  return (dot(N, V))/((dot(N, V)*(1.0f - k) + k));
}

float sqr ( float t ) { return t*t; }
float Fresnel ( float fresnel, float3 wo, float3 H ) {
  fresnel = pow((fresnel-1.0f)/(fresnel+1.0f), 2.0f);
  return fresnel + (1.0f - fresnel)*pow(1.0f - dot(wo, H), 5.0f);
}

float GGX ( float dot_nv, float alpha ) {
  return 2.0f/(1.0f + sqrt(1.0f + alpha*alpha * (1.0f-dot_nv*dot_nv)/(dot_nv*dot_nv)));
}


float hash ( float t ) {
  return fract(sin(t*9372.342432f)*89157453.034813f);
}

float3 Shade_Heightfield ( float3 O, float3 N, float3 wi, float3 sha,
                            float2 info ) {
  // --- colours / fbm ---
  float3 diff = float3(0.529f, 0.356f, 0.005f)*0.05f*IPI;
  float2 fq, fr;
  float pattern = Pattern(u_texture_0, O.xz*2.2f, fq, fr);
  diff += float3(0.0f, 0.1f, 0.0f)*dot(fq.yx, fr.xy*pattern)*0.2f*IPI;
  diff += float3(0.0f, 0.05f, 0.1f)*dot(hash(O.x*0.3f)*fq.xyy*pattern,
                                             fr.yxx-pattern)*0.3f*IPI;
  float3 sun, ambient;
  float3 H;
  // sun
  float tsh = saturate(dot(N, float3(Sun_dir)));
  H = normalize(wi+Sun_dir);
  sun = tsh*diff*float3(1.1f, 1.2f, 1.3f)*Sun_dif;
  sun += (float3(1.0f)
          * (Fresnel(1.5f, Sun_dir, H) + Fresnel(1.5f, wi, H))
          * pow(max(0.0f, dot(N, H)), 5.0f)*((7.0f)/(TAU))
          * pow(dot(N, wi)*dot(N, Sun_dir), 0.5f)
          )*tsh;
  // ambient
  ambient=float3(0.0f);
  float3 refl = float3(0.0f, 1.0f, 0.0f);
  // cubemap
  H = normalize(wi+refl);
  ambient += diff*pow(Cubemap(refl), float3(0.8f))*1.05f;
  // ambient occlusion
  ambient *= Ambient_Occlusion(O);
  return (sun*sha*5.0f+ambient*1.2f);
  // return clamp(dot(N, Sun_dir), 0.0f, 1.0f)*diff;
}

float3 RWater_Sun (  float3 O, float3 N, float3 wi, float3 sha ) {
  // sun
  float3 sun;
  float3 H = normalize(wi+Sun_dir);
  float b;
  b = sqr(dot(N, H));
  b = clamp(exp((b-1.0f)/(0.03f*b))/(PI*0.03f*b*b), 0.0f, 0.1f);
  sun = (float3(1.0f)
      * Fresnel(1.2f, wi, H)
      * GGX(dot(N, wi), 1.01f) * GGX(dot(N, Sun_dir), 0.5f)
      * clamp(
          abs(b * dot(N, Sun_dir)*dot(N, wi)*2.0f/(1.0f+dot(Sun_dir, wi))),
              0.0f, 0.012f)
      * (1.0f/(4.0f*dot(N, wi)*min(dot(N, H), dot(N, Sun_dir))))
      )*Sun_dif;
  return sun;
}

float3 Cheap_Water_Shade ( float3 O, float3 N, float3 wi, float3 sha ) {
  float3 col = float3(0.1f)*IPI;
  float3 sun = RWater_Sun(O, N, wi, sha);
  float3 refl = reflect(-wi, N);
  return sun*sha + Cubemap(refl)*0.01f;
}

float3 Shade_Mirror ( float3 O, float3 N, float3 wi, float3 sha, float2 info ) {
  float3 wo = normalize(Sun_dir);
  float3 refl = reflect(-wi, N);
  float tsh = saturate(dot(N, float3(Sun_dir)));
  return Ambient_Occlusion(O)*float3(0.05f, 0.05f, 0.2f)*
         pow(Cubemap(refl), float3(0.8f));
}

float3 Shade_Water ( float3 O, float3 N, float3 wi, float3 sha,
                            float2 info ) {
  float3 col = float3(0.1f, 0.1f, 0.1f)*IPI;
  float3 sun = RWater_Sun(O, N, wi, sha);
  wi *= -1.0f;
  // refraction
  float3 spec = float3(0.0f);
  float3 refr = refract(wi, N, 0.6f);
  float2 res = March(Ray(O+refr*0.1f, refr));
  if ( res.y == 1.0f ) {
    float3 O = O+refr*0.1f + refr*res.x;
    float3 N = Normal(O, res.x);
    spec += Shade_Heightfield(O, N, refr, float3(0.0f), res)*15.0f;
  }
  // reflection
  float3 refl = reflect(wi, N);
  res = March(Ray(O+refl*0.1f, refl));
  if ( res.x >= 0.0f ) {
    float3 O = O+refl*0.01f + refl*res.x,
            N = Normal(O, res.x);
    if ( res.y == 1.0f )
      spec += Shade_Heightfield(O, N, refl, sha, res);
    if ( res.y == 2.0f )
      spec += Shade_Boid(O, N, refl, sha, res);
  } else spec += Cubemap(float3(refl.x, refl.y, refl.z))*0.01f;
  return (sun*sha+spec);
}

float3 Soft_Shadow ( Ray ray ) {
  float res = 1.0f;
  for ( float t = 0.1f; t < 4.0f; ) {
    float h = Map(ray.ori + ray.dir*t).x;
    if ( h < 0.001f ) return float3(0.00f);
    res = min(res, 2.0f*h/t);
    t += h;
  }
  return pow(float3(saturate(res)), float3(1.02f, 1.2f, 1.5f));
}

void main (  ){
  frag_colour = float4(0.0f, 0.0f, 0.0f, 1.0f);
  float2 uv = frag_coord;
  uv.x *= 1.3333f;
  float3 eye_ori = float3(/*sin(u_time*0.5f)**/8.0f, 13.0f,
                          /*cos(u_time*0.5f)**/8.0f);
  eye_ori += float3(Sun_dir.x, 0.0f, Sun_dir.z)*u_time*4.0f+float3(0.0f, 2.0f, 0.0f);
  float3 eye_target;
  float fov = 1.5f;

  eye_target = float3(Sun_dir.x, -1.5f, Sun_dir.z)*2.0f + eye_ori;
  eye_ori.x += sin(u_time)*-16.0f;
  eye_ori.y += sin(u_time)*4.0f;
  eye_ori.z += cos(u_time)*-16.0f;

    // eye_ori *= (2.0f+u_cam_range)*0.002f;
    // fov += clamp(u_cam_range*0.1f, 0.0f, 3.0f);
    // eye_ori = eye_ori - boid_ori_mass;
    // eye_target = -boid_ori_mass;

  eye_ori.y = max(eye_ori.y, -1.0f);
  Ray eye = Look_At(uv, eye_ori, eye_target,
                        float3(0.0f, 1.0f, 0.0f), fov);
  float2 res = March(eye);
  float dist = res.x;
  float ID = res.y;
  frag_colour.xyz = Cubemap(eye.dir);
  if ( dist <= 0.0 ) {
    return;
  }
  float3 O = eye.ori + eye.dir*dist;
  float3 wi = eye.dir;
  float3 N;
  float3 sun_sha = Soft_Shadow(Ray(O, float3(1.0f, -1.0f, 1.0f)*Sun_dir));
  if ( ID == 3.0f ) N = Water_Normal(O, dist);
  else N = Normal(O, dist);
  float3 col = float3(0.0f);
  wi *= -1.0f;
  if ( ID == 1.0f ) col = Shade_Heightfield(O, N, wi, sun_sha, res);
  if ( ID == 2.0f ) col = Shade_Boid(O, N, wi, sun_sha, res);
  if ( ID == 3.0f ) col = Shade_Water(O, N, wi, sun_sha, res);
  if ( ID == 4.0f ) col = Shade_Mirror(O, N, wi, sun_sha, res);
  // soft shadow
  // col *= sun_sha;
  // col = mix(col, pow(float3(sun_sha), float3(1.0f, 1.2f, 1.5f)), 0.8f);
  // // star
  // float3 L = float3(boid_ori_mass.x, 8.0f, boid_ori_mass.z+2.0f);
  // float3 wo = normalize(L - O);
  // col += float3(0.6f, 0.7f, 0.8f)*dot(N, wo);
  // // sky
  // float3 refl = reflect(wi, N);
  // col += float3(0.6f, 0.7f, 0.8f)*Cubemap(refl)*0.5f;

  // purple fog
  float fog_amount = 1.0f - exp(-dist*0.03f);
  float sun_amount = max(dot(eye.dir, normalize(float3(0.01f, 0.8f, 0.2f))),
                        0.0f);
  float3 fog_colour = mix(float3(0.2f, 0.2f, 0.6f),
                          float3(1.0f, 0.9f, 0.7f), pow(sun_amount, 8.0f));
  // keylight
  // col += fog_colour;
  // col = mix(col, fog_colour, fog_amount);
  frag_colour.xyz = pow(col, float3(1.0f/2.22f));
}
}.format(boidsim.Boid_amt);
