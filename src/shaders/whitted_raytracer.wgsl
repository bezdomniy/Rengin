// NOT WORKING AFTER LOOP SHADER UPDATE - deprecated

@group(0) @binding(0)
var imageData: texture_storage_2d<rgba8unorm,read_write>;
@group(0) @binding(1)
var<uniform> ubo: UBO;
@group(0) @binding(2)
var<storage, read> inner_nodes: InnerNodes;
@group(0) @binding(3)
var<storage, read> leaf_nodes: LeafNodes;
@group(0) @binding(4)
var<storage, read> normal_nodes: Normals;
@group(0) @binding(5)
var<storage, read> object_params: ObjectParams;
@group(0) @binding(6)
var<storage, read_write> rays: array<Ray>;

fn isShadowed(p: vec3<f32>, lightPos: vec3<f32>) -> bool
{
  let v: vec3<f32> = lightPos - p;
  let distance: f32 = length(v);
  let direction: vec3<f32> = normalize(v);

  let intersection: Intersection = intersect(Ray(p,0,direction,0));

  if (intersection.closestT > EPSILON && intersection.closestT < distance)
  {
    return true;
  }

  return false;
}


fn lighting(material: Material, lightPos: vec3<f32>, light_emissiveness: vec4<f32>, hitParams: HitParams, shadowed: bool) -> vec4<f32>
{
  // return material.colour;
  var diffuse: vec4<f32>;
  var specular: vec4<f32>;

//   let intensity = vec4<f32>(1.0,1.0,1.0,1.0); // TODO temp placeholder

  let effectiveColour = light_emissiveness * material.colour; //* light->intensity;

  let ambient = effectiveColour * material.ambient;
  // vec4 ambient = vec4(0.3,0.0,0.0,1.0);
  if (shadowed) {
    return ambient;
  }

  let lightv: vec3<f32> = normalize(lightPos - hitParams.overPoint);

  let lightDotNormal: f32 = dot(lightv, hitParams.normalv);
  if (lightDotNormal < 0.0)
  {
    diffuse = vec4<f32>(0.0, 0.0, 0.0,0.0);
    specular = vec4<f32>(0.0, 0.0, 0.0,0.0);
  }
  else
  {
    // compute the diffuse contribution​
    diffuse = effectiveColour * material.diffuse * lightDotNormal;

    // reflect_dot_eye represents the cosine of the angle between the
    // reflection vector and the eye vector. A negative number means the
    // light reflects away from the eye.​
    let reflectv = reflect(-lightv, hitParams.normalv);
    let reflectDotEye = dot(reflectv, hitParams.eyev);

    if (reflectDotEye <= 0.0)
    {
      specular = vec4<f32>(0.0, 0.0, 0.0,0.0);
    }
    else
    {
      // compute the specular contribution​
      let factor = pow(reflectDotEye, material.shininess);
      specular = light_emissiveness * material.specular * factor;
    }
  }

  return (ambient + diffuse + specular);
}


fn renderScene(init_ray: Ray) -> vec4<f32> {
    // int id = 0;
    var color: vec4<f32> = vec4<f32>(0.0);
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // var ray: Ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);

    // let init_ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);
    // let init_ray = rays.Rays[(pixel.y * ubo.width) + pixel.x];
    var type_enum = 0;
    // var intersection: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));

    var stack: array<Ray,MAX_RAY_BOUNCE_ARRAY_SIZE>;
    var top_stack = -1;

    top_stack = top_stack + 1;
    stack[top_stack] = init_ray;

    // TODO: check this is light model_type (9), currently not compatible with pathtracer scenes
    let light = object_params.ObjectParams[ubo.lights_offset];
    // inverse_transform for a light is just the transform (not inversed)
    let light_position = light.transform * vec4<f32>(0.0,0.0,0.0,1.0);
    let light_emissiveness = light.material.emissiveness;

    loop  {
        if (top_stack < 0) { break; }

        let new_ray = stack[top_stack];
        top_stack = top_stack - 1;

        if (new_ray.bounce_number >= ubo.ray_bounces) { 
            continue;
        }

        let intersection = intersect(new_ray.ray);
        
        if (intersection.closestT >= MAXLEN || intersection.id == -1)
        {
            continue;
        }

        // TODO: just hard code object type in the intersection rather than looking it up
        let ob_params = object_params.ObjectParams[intersection.model_id];


        let hitParams: HitParams = getHitParams(new_ray.ray, intersection, ob_params.model_type);
        let shadowed: bool = isShadowed(hitParams.overPoint, light_position.xyz);
        // let shadowed = false;
        
        let albedo = lighting(ob_params.material,
                                light_position.xyz,
                                light_emissiveness,
                                hitParams, shadowed) 
                                * new_ray.reflectance 
                                * new_ray.reflective 
                                * new_ray.transparent
                                ;
        color = color + albedo;

        if (ob_params.material.transparency > 0.0 || ob_params.material.reflective > 0.0) {
            var eta_t = ob_params.material.refractive_index;

            var cos_i = clamp(-1.0,1.0,dot(hitParams.eyev,hitParams.normalv));

            if (!hitParams.front_face) {
                eta_t=eta_t/new_ray.refractive_index;
            }

            var reflectance = 1.0;
            let do_schlick = ob_params.material.transparency > 0.0 && ob_params.material.reflective > 0.0;

            let n_ratio = new_ray.refractive_index / eta_t;
            let sin_2t = pow(n_ratio,2.0) * (1.0 - pow(cos_i,2.0));
            let cos_t = sqrt(1.0 - sin_2t);

            if (do_schlick) {
                reflectance = schlick(cos_i, eta_t, new_ray.refractive_index);
            }

            if (ob_params.material.reflective > 0.0) {
                top_stack = top_stack + 1;
                stack[top_stack] = RenderRay (Ray(hitParams.overPoint, new_ray.ray.x, hitParams.reflectv,new_ray.ray.y),new_ray.bounce_number + 1u,reflectance * new_ray.reflectance,new_ray.reflective * ob_params.material.reflective,1.0,ob_params.material.refractive_index); 
            }
                        
            if (sin_2t <= 1.0 && ob_params.material.transparency > 0.0) {
                let direction = hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                            (hitParams.eyev * n_ratio);

                top_stack = top_stack + 1;
                stack[top_stack] = RenderRay (Ray(hitParams.underPoint,new_ray.ray.x, direction,new_ray.ray.y),new_ray.bounce_number + 1u,(1.0-reflectance) * new_ray.reflectance,1.0,new_ray.transparent * ob_params.material.transparency,ob_params.material.refractive_index); 
           
            }
        }
    }
 
    return color;
}


@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) 
{
    let offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    let ray = rays[offset];

    if (ray.bounce_idx < 0) {
        return;
    }

    //TODO: why this needed
    if (ubo.ray_bounces == u32(ray.bounce_idx + 1)) {
        var color: vec4<f32> = RAY_MISS_COLOUR;
        if (ubo.subpixel_idx > 0u) {
            color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
        }
        let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

        color = mix(color,vec4<f32>(0.0,0.0,0.0,1.0),scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
        return;
    }

    let ray_color = renderScene(ray,offset,light_sample);

    var color: vec4<f32> = RAY_MISS_COLOUR;
    if (ubo.subpixel_idx > 0u) {
        color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
    }
    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    if (ray_color.w > -EPSILON) {
        color = mix(color,ray_color,scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
    }
}
