module boidsim;
import derelict.opengl, derelict.glfw3;
import globals;

// Render boids
struct Boid {
  float3 ori, dir;
  float velocity;
  float timer, timer_max;
  float3 uniform_dir_0, uniform_dir_1;
};
immutable size_t Boid_amt = 7;
Boid[Boid_amt] boids;
float3 ori_mass, nor_flock_dir;
float cam_range;

void Initialize() {
  foreach ( iter, ref boid; boids ) {
    boid.ori = float3(sin(iter*2.5f), -5.0f, cos(iter*2.5f))*0.5f;
    boid.dir = normalize(float3(0.0f, 0.0f, 1.0f));
    boid.velocity = 0.10f;
    boid.timer = boid.timer_max = 0.0f;
  }
}

Boid* Find_Nearest(size_t iter) {
  Boid* nearest_boid;
  float nearest_dist = float.max;
  foreach ( titer, ref tboid; boids ) {
    float ndist = distance(boids[iter].ori, tboid.ori);
    if ( titer != iter && ndist < nearest_dist ) {
      nearest_boid = &boids[titer];
      nearest_dist = ndist;
    }
  }
  return nearest_boid;
}

float3 Calculate_Random(size_t iter, float delta) {
  Boid* boid = &boids[iter];
  boid.timer += delta;
  if ( boid.timer >= boid.timer_max ) {
    boid.timer = 0.0f;
    boid.timer_max = uniform(2.0f, 10.0f);
    boid.uniform_dir_0 = Sample_Float3()*uniform(0.5f, 1.5f);
    boid.uniform_dir_1 = Sample_Float3()*uniform(0.5f, 1.5f);
  }
  float3 T = Mix(boid.uniform_dir_0, boid.uniform_dir_1,
                    boid.timer/boid.timer_max);
  return T;
}

float3 Calculate_Seperation(size_t iter) {
  Boid boid = boids[iter];
  float3 velocity = float3(0.0f);
  foreach ( titer, tboid; boids ) {
    if ( titer == iter ) continue;
    float dist = max(distance(boid.ori, tboid.ori), 1.0f);
    velocity += normalize(boid.ori - tboid.ori)/dist;
  }
  return normalize(velocity) - boid.velocity*boid.dir;
}

float3 Calculate_Alignment(size_t iter) {
  Boid boid = boids[iter];
  float3 velocity = float3(0.0f);
  int count = 0;
  foreach ( titer, tboid; boids ) {
    if ( titer == iter ) continue;
    ++ count;
    float d = max(distance(boid.ori, tboid.ori), 1.0f);
    velocity += tboid.dir*tboid.velocity;
  }

  return normalize(velocity/cast(float)count) - boid.velocity*boid.dir;
}

float3 Calculate_Cohesion(size_t iter) {
  Boid boid = boids[iter];
  float3 ori = float3(0.0f);
  int count = 0;
  foreach ( titer, tboid; boids ) {
    if ( titer == iter ) continue;
    ori += tboid.ori;
    ++ count;
  }
  return normalize((ori/cast(float)count) - boid.ori)-boid.velocity*boid.dir;
}

void Boid_Iterate ( float u_time, float delta ) {
  // find center of mass and normalized flocking direction
  ori_mass = boids[].Truncate!"ori".OpReduce!"+"/cast(float)Boid_amt;
  nor_flock_dir = normalize(boids.Truncate!"dir".OpReduce!"+"/cast(float)Boid_amt);
  // apply direction vectors
  foreach ( iter, ref boid; boids ) {
    float3 ori = boid.ori, dir = boid.dir;
    auto nearest_boid = Find_Nearest(iter);
    float3 ndir = normalize(ori - nearest_boid.dir);
    float3 accel = float3(0.0);
    // seperators
    accel += iter.Calculate_Seperation*1.4f;
    // // cohesions
    accel += iter.Calculate_Alignment*0.2f;
    accel += iter.Calculate_Cohesion*1.6f;
    // randoms
    accel += iter.Calculate_Random(delta)*0.4f;
    // accel += ori.y/10.0f;
    // leader
    if ( iter == 0 ) { // run away
      accel += normalize(Mix(iter.Calculate_Alignment*0.5f,
                   iter.Calculate_Random(delta)*0.4f, 0.3f))*0.7f;
      accel += iter.Calculate_Seperation*1.4f;
      if ( ori.y <= -3.0f ) accel.y += 2.5f;
      if ( ori.y >= 1.0f ) accel.y -= 1.0f;
      if ( boid.timer <= 0.5f ) boid.velocity *= 0.998f;
    } else {
      auto l = boids[0];
      float dist = clamp(distance(l.ori, ori), 1.0f, 3.0f);
      accel += normalize(l.ori - ori)*dist*0.2f;
    }
    // bounds check
    if ( ori.y >= 1.3f ) accel.y -= 0.5f;
    // add up and stuff
    boid.velocity *= 0.9991f;
    float3 tvel = boid.velocity*boid.dir + (accel)*0.001f;
    tvel.y = clamp(tvel.y, -0.05f, 0.05f);
    // boid.velocity = clamp(tvel.length, -0.1f, 0.1f);
    boid.dir = normalize(tvel);
  }
  // apply origin/velocity
  foreach ( ref boid; boids ) {
    boid.ori += boid.dir*boid.velocity;
  }
  // get camera range
  float3 minbbox = float3(float.max),
         maxbbox = float3(-float.max);
  foreach ( ref boid; boids ) {
    if ( boid.ori.x < minbbox.x ) minbbox.x = boid.ori.x;
    if ( boid.ori.x > maxbbox.x ) maxbbox.x = boid.ori.x;
    if ( boid.ori.y < minbbox.y ) minbbox.y = boid.ori.y;
    if ( boid.ori.y > maxbbox.y ) maxbbox.y = boid.ori.y;
    if ( boid.ori.z < minbbox.z ) minbbox.z = boid.ori.z;
    if ( boid.ori.z > maxbbox.z ) maxbbox.z = boid.ori.z;
  }
  cam_range = distance(minbbox, maxbbox);
}

