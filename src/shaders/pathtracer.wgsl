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


fn onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;
    out[2] = n;

    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign / n.z);
    let b = n.x * n.y * a;
    out[0] = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    out[1] = vec3<f32>(b, sign + n.y * n.y * a, -n.y);

    return out;
}

fn _onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;

    out[2] = n;
    if n.z < -0.99995 {
        out[0] = vec3<f32>(0.0, -1.0, 0.0);
        out[1] = vec3<f32>(-1.0, 0.0, 0.0);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        out[0] = vec3<f32>(1.0 - n.x * n.x * a, b, -n.x);
        out[1] = vec3<f32>(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return out;
}


// fn onb(n: vec3<f32>) -> array<vec3<f32>,3> {
//     var out: array<vec3<f32>,3>;
//     out[2] = n;

//     if (abs(out[2].x) > 0.9) {
//         let a = vec3<f32>(0.0,1.0,0.0);
//     }
//     else {
//         let a = vec3<f32>(1.0,0.0,0.0);
//     }
//     out[1] = normalize(cross(out[2], a));
//     out[0] = cross(out[2], out[1]);

//     return out;
// }

fn onb_local(v: vec3<f32>, onb: mat3x3<f32>) -> vec3<f32> {
    // return v.x*onb[0] + v.y*onb[1] + v.z*onb[2];
    return onb * v;
}


// TODO: check the initial size cube and sphere and centroid coordinates and adjust accordingly
fn surface_area(object: ObjectParam) -> f32 {
    let scale = vec3<f32>(length(object.transform[0].xyz), length(object.transform[1].xyz), length(object.transform[2].xyz));
    if object.model_type == 0u {
        return 4.0 * PI * pow((pow(scale.x * scale.z, 1.6075) + pow(scale.z * scale.y, 1.6075) + pow(scale.x * scale.y, 1.6075)) / 3.0, 1.0 / 1.6075);
    } else if object.model_type == 1u {
        // TODO
        return 1.0;
    } else if object.model_type == 2u {
        // return (343.0 - 213.0) * (332.0 - 227.0);
        return 2.0 * ((2.0 * scale.x * scale.z) + (2.0 * scale.z * scale.y) + (2.0 * scale.x * scale.y));
    } else if object.model_type == 9u {
        return 1.0;
    }

    // TODO
    return 1.0;
}

fn light_pdf(ray: Ray, intersection: Intersection) -> f32 {
    if intersection.id == -1 || intersection.closestT >= MAXLEN {
        return 0f;
    }

    let light = object_params.ObjectParams[intersection.model_id];

    if light.model_type == 0u {
        let scale = vec3<f32>(length(light.transform[0].xyz), length(light.transform[1].xyz), length(light.transform[2].xyz));
        let radius = max(max(scale.x, scale.y), scale.z);

        // let hit_point = ray.rayO + intersection.closestT * normalize(ray.rayD);
        // let normal = normalAt(hit_point, intersection, 0u);
        // let center = hit_point - (normal * radius );
        let center = (light.transform * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

        let cos_theta_max = sqrt(1.0 - (radius * radius / pow(length(center - ray.rayO), 2.0)));
        let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
        return 1.0 / solid_angle;


        // let cos_theta = dot(-ray.rayD, normalize(hit_point - center));
        // let light_area = surface_area(light);
        // return pow(intersection.closestT,2.0) / (light_area * cos_theta * 0.5);


        // let light_area = surface_area(light);
        // return pow(intersection.closestT,2.0) / (cosine * light_area);

        // let light_area = surface_area(light);
        // let point = ray.rayO + normalize(ray.rayD) * intersection.closestT;
        // let n_light = normalAt(point, intersection,0u);
        // let solid_angle = (light_area * dot(n_light, ray.rayD)) / pow(intersection.closestT,2.0);
        // return 1.0 / solid_angle;
    } else if light.model_type == 1u {
        // TODO
        return 1.0;
    } else if light.model_type == 2u {
        let p = ray.rayO + normalize(ray.rayD) * intersection.closestT;
        let normal = normalAt(p, intersection, light.model_type);
        let light_area = surface_area(light);
        // return 1f - light_area;
        let distance_squared = pow(intersection.closestT, 2f);
        let cosine = abs(dot(normal, -ray.rayD));

        return distance_squared / (cosine * light_area);
    } else if light.model_type == 9u {
        return 1.0;
    }

    // TODO
    return 1.0;
}

fn renderScene(ray: Ray, offset: u32, light_sample: bool) -> vec4<f32> {
    var p_scatter = 0.5;
    if !light_sample || ubo.lights_offset == ubo.n_objects {
        p_scatter = 1.0;
    }
        
    // Get intersected object ID
    let intersection = intersect(ray, 0u, false);

    if intersection.id == -1 || intersection.closestT >= MAXLEN {
        rays[offset] = Ray(vec3<f32>(-1f), -1f, vec3<f32>(-1f), -1, vec4<f32>(-1f));
        return ray.throughput * RAY_MISS_COLOUR;
    }

    // TODO: just hard code object type in the intersection rather than looking it up
    let ob_params = object_params.ObjectParams[intersection.model_id];

    if ob_params.material.emissiveness.x > 0.0 {
        rays[offset] = Ray(vec3<f32>(-1f), -1f, vec3<f32>(-1f), -1, vec4<f32>(-1f));
        return ray.throughput * ob_params.material.emissiveness;
    }
    let hitParams = getHitParams(ray, intersection, ob_params.model_type);

    let is_specular = ob_params.material.reflective > 0.0 || ob_params.material.transparency > 0.0;
    var scattering_target = vec3<f32>(0.0);
    var p = hitParams.overPoint;

    let onb = onb(hitParams.normalv);

    if is_specular {
        if ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0 {
            scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective));
        } else {
            var eta_t = ob_params.material.refractive_index;

            let cos_i = min(dot(hitParams.eyev, hitParams.normalv), 1.0);

            if !hitParams.front_face {
                eta_t = eta_t / ray.refractive_index;
            }

            // let reflectance = schlick(cos_i, eta_t, ray.refractive_index);
            let reflectance = schlick_lazanyi(cos_i, eta_t, 0.0);

            let n_ratio = ray.refractive_index / eta_t;
            let sin_2t = pow(n_ratio, 2.0) * (1.0 - pow(cos_i, 2.0));

            if sin_2t > 1.0 || reflectance >= f32_zero_to_one(rand_pcg4d.w) {
                // scattering_target = hitParams.reflectv;
                // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * random_uniform_direction());
                scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective));
                // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * hemisphericalRand(1.0,hitParams.normalv));
            } else {
                let cos_t = sqrt(1.0 - sin_2t);

                // scattering_target = (hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                //                 (hitParams.eyev * n_ratio)) + ((1.0 - ob_params.material.transparency) * onb_local(-random_cosine_direction(), onb));
                scattering_target = hitParams.normalv * ((n_ratio * cos_i) - cos_t) - (hitParams.eyev * n_ratio);
                p = hitParams.underPoint;
            }
        }

        rays[offset] = Ray(p, ob_params.material.refractive_index, normalize(scattering_target), ray.bounce_idx + 1, ray.throughput * ob_params.material.colour);
    } else {
        if light_sample && f32_zero_to_one(rand_pcg4d.w) < p_scatter {
            let light = random_light();
            scattering_target = random_to_light(light, p);
        } else {
            // scattering_target = random_cosine_direction();
            scattering_target = onb * random_cosine_direction();
        }
        let direction = normalize(scattering_target);

        let scattering_cosine = dot(direction, onb[2]);

        let scattering_pdf = max(0f, scattering_cosine / PI);

        var v_light_pdf = 0f;

        var next_ray = Ray(p, ob_params.material.refractive_index, direction, ray.bounce_idx + 1, vec4<f32>(-1f));

        for (var i_light: u32 = ubo.lights_offset; i_light < ubo.n_objects; i_light = i_light + 1u) {
            let l_intersection = intersect(next_ray, i_light, true);
            if l_intersection.id == -1 || l_intersection.closestT >= MAXLEN {
                continue;
            }

            v_light_pdf = v_light_pdf + light_pdf(next_ray, l_intersection);
        }

        v_light_pdf = v_light_pdf / f32(ubo.n_objects - ubo.lights_offset);
        let pdf = (p_scatter * scattering_pdf) + ((1.0 - p_scatter) * v_light_pdf);

        next_ray.throughput = ray.throughput * ob_params.material.colour * scattering_pdf / pdf;

        rays[offset] = next_ray;
    }

    return vec4<f32>(-1f);
}


@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    let ray = rays[offset];

    if ray.bounce_idx < 0 {
        return;
    }

    //TODO: why this needed
    if ubo.ray_bounces == u32(ray.bounce_idx + 1) {
        var color: vec4<f32> = RAY_MISS_COLOUR;
        if ubo.subpixel_idx > 0u {
            color = textureLoad(imageData, vec2<i32>(global_invocation_id.xy));
        }
        let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

        color = mix(color, vec4<f32>(0.0, 0.0, 0.0, 1.0), scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
        return;
    }

    var light_sample = true;
    if ubo.lights_offset == ubo.n_objects {
        light_sample = false;
    }

    init_pcg4d(vec4<u32>(global_invocation_id.x, global_invocation_id.y, ubo.subpixel_idx, u32(ray.bounce_idx)));
    let ray_color = renderScene(ray, offset, light_sample);

    var color: vec4<f32> = RAY_MISS_COLOUR;
    if ubo.subpixel_idx > 0u {
        color = textureLoad(imageData, vec2<i32>(global_invocation_id.xy));
    }
    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    if ray_color.w > -EPSILON {
        color = mix(color, ray_color, scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
    }
}
