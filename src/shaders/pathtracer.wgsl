#import rengin::types::{ObjectParam, Intersection, HitParams, Ray}
#import rengin::constants::{EPSILON, MAXLEN, PI, RAY_MISS_COLOUR}
#import rengin::random::{
    init_pcg3d, next_pcg3d, rand_pcg3d,
    f32_zero_to_one, f32_negone_to_one, rescale,
    random_cosine_direction, random_to_sphere,
}
#import rengin::pt_bindings::{
    imageData, ubo, object_params, ray_throughputs,
    active_rays_in, active_rays_out, ray_counts,
    load_ray, store_ray, store_dead_ray,
}
#import rengin::stats::flush_stats
#import rengin::hit_params::{getHitParams, schlick_lazanyi}
#import rengin::intersects::intersect


// // Buggy
// fn onb(n: vec3<f32>) -> mat3x3<f32> {
//     var out: mat3x3<f32>;
//     out[2] = n;

//     let sign = select(-1.0, 1.0, n.z >= 0.0);
//     let a = -1.0 / (sign / n.z);
//     let b = n.x * n.y * a;
//     out[0] = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
//     out[1] = vec3<f32>(b, sign + n.y * n.y * a, -n.y);

//     return out;
// }

fn onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;
    out[2] = n;

    var a = vec3<f32>(1.0, 0.0, 0.0);
    if abs(out[2].x) > 0.9 {
        a = vec3<f32>(0.0, 1.0, 0.0);
    }
    out[1] = normalize(cross(out[2], a));
    out[0] = cross(out[2], out[1]);

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

fn random_light() -> ObjectParam {
    next_pcg3d();
    let i = u32(rescale(f32_zero_to_one(rand_pcg3d.x), f32(ubo.lights_offset), f32(ubo.n_objects)));
    return object_params.ObjectParams[i];
}

fn random_to_light(light: ObjectParam, origin: vec3<f32>) -> vec3<f32> {
    let center = light.transform[3].xyz;
    let direction = center - origin;
    let distance_squared = dot(direction, direction);

    let onb = onb(normalize(direction));

    let scale = vec3<f32>(length(light.transform[0].xyz), length(light.transform[1].xyz), length(light.transform[2].xyz));


    if light.model_type == 0u {
        let radius = max(max(scale.x, scale.y), scale.z);
        // let radius = 1.0;
        let r = onb * random_to_sphere(radius, distance_squared);
        return r;
    } else if light.model_type == 1u {
        return vec3<f32>(0.0);
    } else if light.model_type == 2u {
        // Area-weighted face sampling for non-uniform cubes
        let area_yz = scale.y * scale.z;  // x-face area (proportional)
        let area_xz = scale.x * scale.z;  // y-face area
        let area_xy = scale.x * scale.y;  // z-face area
        let total = 2.0 * (area_yz + area_xz + area_xy);

        next_pcg3d();
        let u = f32_zero_to_one(rand_pcg3d.x);
        let v = f32_negone_to_one(rand_pcg3d.y);
        let w = f32_negone_to_one(rand_pcg3d.z);
        let face_rand = u * total;

        var p: vec3<f32>;
        if face_rand < area_yz {
            p = vec3<f32>(-1.0, v, w);
        } else if face_rand < 2.0 * area_yz {
            p = vec3<f32>(1.0, v, w);
        } else if face_rand < 2.0 * area_yz + area_xz {
            p = vec3<f32>(v, -1.0, w);
        } else if face_rand < 2.0 * area_yz + 2.0 * area_xz {
            p = vec3<f32>(v, 1.0, w);
        } else if face_rand < 2.0 * area_yz + 2.0 * area_xz + area_xy {
            p = vec3<f32>(v, w, -1.0);
        } else {
            p = vec3<f32>(v, w, 1.0);
        }
        let r = (light.transform * vec4<f32>(p, 1f)).xyz;
        return normalize(r - origin);
    }

    // TODO: for model mesh, choose random triangle, then random point on it
    return vec3<f32>(1.0);
}

fn light_pdf(ray: Ray, intersection: Intersection) -> f32 {
    let light = object_params.ObjectParams[intersection.model_id];

    if light.model_type == 0u {
        let centre = light.transform[3].xyz;
        let scale = vec3<f32>(length(light.transform[0].xyz), length(light.transform[1].xyz), length(light.transform[2].xyz));
        let radius = max(scale.x, max(scale.y, scale.z));
        let length = length(centre - ray.rayO);
        let cos_theta_max = sqrt(1f - radius * radius / (length * length));
        let solid_angle = 2f * PI * (1f - cos_theta_max);
        return  1f / solid_angle;
    } else if light.model_type == 1u {
        // TODO
        return 1.0;
    } else if light.model_type == 2u {
        // Use bounding sphere solid angle (numerically stable, no grazing-angle singularity)
        let centre = light.transform[3].xyz;
        let scale = vec3<f32>(length(light.transform[0].xyz), length(light.transform[1].xyz), length(light.transform[2].xyz));
        let radius = max(scale.x, max(scale.y, scale.z));
        let len = length(centre - ray.rayO);
        let cos_theta_max = sqrt(1f - radius * radius / (len * len));
        let solid_angle = 2f * PI * (1f - cos_theta_max);
        return  1f / solid_angle;
    }

    // TODO
    return 1.0;
}

fn renderScene(in_ray: Ray, offset: u32, light_sample: bool) -> vec4<f32> {
    var ray = in_ray;

    // Russian roulette (pbrt-style): probabilistically terminate low-throughput paths
    // Only apply after depth > 1 to preserve initial bounces unconditionally
    let max_component = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));
    if max_component < 1.0 {
        // Cap at 0.5: the MIS one-sample estimator can produce very low throughput
        // for light-sampled paths (scattering_pdf/pdf << 1) which are still valuable.
        // Unlike pbrt where BSDF*cos/pdf <= 1, our MIS weight spans orders of magnitude.
        let rr_q = min(0.5, max(0.0, 1.0 - max_component));
        next_pcg3d();
        if f32_zero_to_one(rand_pcg3d.z) < rr_q {
            store_dead_ray(offset);
            return ray.throughput * RAY_MISS_COLOUR;
        }
        ray.throughput = vec4<f32>(ray.throughput.xyz / (1.0 - rr_q), ray.throughput.w);
    }

    // Get intersected object ID
    let intersection = intersect(ray, 0u, false);

    if intersection.id == -1 || intersection.closestT >= MAXLEN {
        store_dead_ray(offset);
        return ray.throughput * RAY_MISS_COLOUR;
    }

    // TODO: just hard code object type in the intersection rather than looking it up
    let ob_params = object_params.ObjectParams[intersection.model_id];

    if any(ob_params.material.emissiveness.xyz > vec3<f32>(0.0)) {
        store_dead_ray(offset);
        return ray.throughput * ob_params.material.emissiveness;
    }
    let hitParams = getHitParams(ray, intersection, ob_params.model_type);

    let is_specular = ob_params.material.reflective > 0.0 || ob_params.material.transparency > 0.0;
    var direction = vec3<f32>(0.0);
    var p = hitParams.overPoint;

    let onb = onb(hitParams.normalv);

    if is_specular {
        // Track the medium the ray will be in after this interaction
        var next_ri = ray.refractive_index;

        // Path regularization (pbrt 13.1.4): after non-specular interactions,
        // increase roughness to reduce variance from indirect caustics.
        // throughput < 1 indicates the path has been through at least one attenuating bounce.
        var roughness = 1.0 - ob_params.material.reflective;
        if max_component < 0.99 {
            roughness = clamp(roughness * 2.0, 0.1, 0.5);
        }

        if ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0 {
            direction = hitParams.reflectv + (roughness * onb * random_cosine_direction());
            // Reflection stays in the same medium
        } else {
            var eta_t = ob_params.material.refractive_index;

            let cos_i = min(dot(hitParams.eyev, hitParams.normalv), 1.0);

            if !hitParams.front_face {
                if ray.refractive_index > 1.0 + EPSILON {
                    // Ray is inside a dense medium (glass) -> exiting to air
                    eta_t = 1.0;
                }
                // else: ray is in air (RI~1.0) hitting back face -> entering this material
                // eta_t stays as material RI
            }

            let reflectance = schlick_lazanyi(cos_i, eta_t, 0.0);

            let n_ratio = ray.refractive_index / eta_t;
            let sin_2t = pow(n_ratio, 2.0) * (1.0 - pow(cos_i, 2.0));

            next_pcg3d();
            if sin_2t > 1.0 || reflectance >= f32_zero_to_one(rand_pcg3d.y) {
                direction = hitParams.reflectv + (roughness * onb * random_cosine_direction());
                // Reflection stays in the same medium
            } else {
                let cos_t = sqrt(1.0 - sin_2t);
                direction = hitParams.normalv * ((n_ratio * cos_i) - cos_t) - (hitParams.eyev * n_ratio);
                p = hitParams.underPoint;
                // Refraction enters the transmitted medium
                next_ri = eta_t;
            }
        }

        var specular_weight = ob_params.material.colour;
        if ob_params.material.reflective > 0.0 && ob_params.material.reflective < 1.0 {
            specular_weight = specular_weight * ob_params.material.reflective;
        }
        store_ray(offset, Ray(p, next_ri, normalize(direction), ray.pos, ray.throughput * specular_weight));
    } else {
        if light_sample {
            let p_scatter = 0.5;
            next_pcg3d();
            if f32_zero_to_one(rand_pcg3d.y) < p_scatter {
                let light = random_light();
                direction = normalize(random_to_light(light, p));
            } else {
            // scattering_target = random_cosine_direction();
                let scattering_target = onb * random_cosine_direction();
                direction = normalize(scattering_target);
            }

            var next_ray = Ray(p, ob_params.material.refractive_index, direction, ray.pos, vec4<f32>(-1f));

            let scattering_cosine = dot(direction, onb[2]);
            let scattering_pdf = max(0f, scattering_cosine / PI);
            var v_light_pdf = 0f;

            for (var i_light: u32 = ubo.lights_offset; i_light < ubo.n_objects; i_light = i_light + 1u) {
                let l_intersection = intersect(next_ray, i_light, true);
                if l_intersection.model_id != i_light || l_intersection.id == -1 || l_intersection.closestT >= MAXLEN {
                    continue;
                }

                v_light_pdf = v_light_pdf + light_pdf(next_ray, l_intersection);
            }

            v_light_pdf = v_light_pdf / f32(ubo.n_objects - ubo.lights_offset);
            let pdf = (p_scatter * scattering_pdf) + ((1.0 - p_scatter) * v_light_pdf);

            next_ray.throughput = ray.throughput * ob_params.material.colour * scattering_pdf / pdf;

            store_ray(offset, next_ray);
        } else {
            let scattering_target = onb * random_cosine_direction();
            direction = normalize(scattering_target);
            store_ray(offset, Ray(p, ob_params.material.refractive_index, direction, ray.pos, ray.throughput * ob_params.material.colour));
        }
    }

    return vec4<f32>(-1f);
}


fn process_bounce(offset: u32, pixel_pos: vec2<u32>) {
    let ray = load_ray(offset);

    if ray.throughput.w < -EPSILON {
        flush_stats(false);
        return;
    }

    var color: vec4<f32> = RAY_MISS_COLOUR;
    if ubo.subpixel_idx > 0u {
        color = textureLoad(imageData, vec2<i32>(pixel_pos));
    }
    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    var light_sample = true;
    if ubo.lights_offset == ubo.n_objects {
        light_sample = false;
    }

    init_pcg3d(vec3<u32>(bitcast<u32>(ray.rayD.x), bitcast<u32>(ray.rayD.y), bitcast<u32>(ray.rayD.z)));

    let ray_color = renderScene(ray, offset, light_sample);

    if ray_color.w > -EPSILON {
        let clamped_color = clamp(ray_color, vec4<f32>(0.0), vec4<f32>(3.0));
        color = mix(color, clamped_color, scale);
        textureStore(imageData, vec2<i32>(pixel_pos), color);
    } else if ubo.subpixel_idx == 0u {
        textureStore(imageData, vec2<i32>(pixel_pos), color);
    }

    // If ray is still alive after this bounce, write to compaction output
    let updated_throughput = ray_throughputs[offset];
    if updated_throughput.w >= -EPSILON {
        let out_idx = atomicAdd(&ray_counts[1], 1u);
        active_rays_out[out_idx] = offset;
    }

    flush_stats(true);
}

// Entry point for first bounce: 2D dispatch, all pixels, preserves spatial coherence
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    process_bounce(offset, global_invocation_id.xy);
}

// Entry point for subsequent bounces: 1D compacted dispatch, only alive rays
@compute @workgroup_size(256)
fn main_compact(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let compact_idx = global_invocation_id.x;
    if compact_idx >= atomicLoad(&ray_counts[0]) {
        return;
    }
    let offset = active_rays_in[compact_idx];
    let px = offset % ubo.resolution.x;
    let py = offset / ubo.resolution.x;
    process_bounce(offset, vec2<u32>(px, py));
}
