#import rengin::types::{ObjectParam, Intersection, HitParams, Ray}
#import rengin::constants::{EPSILON, MAXLEN, PI, RAY_MISS_COLOUR}
#import rengin::random::{
    init_pcg3d, next_pcg3d, rand_pcg3d,
    f32_zero_to_one, f32_negone_to_one, rescale,
    random_cosine_direction, random_to_sphere,
}
#import rengin::pt_bindings::{
    imageData, ubo, object_params,
    active_rays_in, active_rays_out, ray_counts,
    load_ray, store_ray, store_dead_ray, load_ray_throughput,
}
#import rengin::stats::flush_stats
#import rengin::hit_params::getHitParams
#import rengin::fresnel::schlick_lazanyi
#import rengin::intersects::intersect
#import rengin::pathtracer_utils::{onb, surface_area, onb_local}
#import rengin::pathtracer_math::{
    light_pdf_sphere, scattering_pdf as compute_scattering_pdf,
    mis_pdf, snell_refract,
    russian_roulette_q, russian_roulette_adjust,
    specular_weight as compute_specular_weight,
    path_regularization_roughness,
    determine_eta_t, cube_face_select,
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
        next_pcg3d();
        let u = f32_zero_to_one(rand_pcg3d.x);
        let v = f32_negone_to_one(rand_pcg3d.y);
        let w = f32_negone_to_one(rand_pcg3d.z);

        let face = cube_face_select(scale, u);
        var p: vec3<f32>;
        switch face {
            case 0u { p = vec3<f32>(-1.0, v, w); }
            case 1u { p = vec3<f32>(1.0, v, w); }
            case 2u { p = vec3<f32>(v, -1.0, w); }
            case 3u { p = vec3<f32>(v, 1.0, w); }
            case 4u { p = vec3<f32>(v, w, -1.0); }
            default  { p = vec3<f32>(v, w, 1.0); }
        }
        let r = (light.transform * vec4<f32>(p, 1f)).xyz;
        return normalize(r - origin);
    }

    // TODO: for model mesh, choose random triangle, then random point on it
    return vec3<f32>(1.0);
}

fn light_pdf(ray: Ray, intersection: Intersection) -> f32 {
    let light = object_params.ObjectParams[intersection.model_id];

    if light.model_type == 0u || light.model_type == 2u {
        // Sphere and cube both use bounding-sphere solid angle PDF
        let centre = light.transform[3].xyz;
        let scale = vec3<f32>(length(light.transform[0].xyz), length(light.transform[1].xyz), length(light.transform[2].xyz));
        let radius = max(scale.x, max(scale.y, scale.z));
        let dist = length(centre - ray.rayO);
        return light_pdf_sphere(radius, dist);
    } else if light.model_type == 1u {
        // TODO
        return 1.0;
    }

    // TODO
    return 1.0;
}

fn renderScene(in_ray: Ray, offset: u32, light_sample: bool) -> vec4<f32> {
    var ray = in_ray;

    // Russian roulette disabled: throughput alone can't distinguish deep paths from
    // caustic paths with low MIS weights. Needs explicit bounce depth tracking (TODO).
    // Path termination is handled by the bounded bounce dispatch loop.
    let max_component = max(ray.throughput.x, max(ray.throughput.y, ray.throughput.z));

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
        var roughness = path_regularization_roughness(ob_params.material.reflective, max_component);

        if ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0 {
            direction = hitParams.reflectv + (roughness * onb * random_cosine_direction());
            // Reflection stays in the same medium
        } else {
            let cos_i = min(dot(hitParams.eyev, hitParams.normalv), 1.0);
            let eta_t = determine_eta_t(ray.refractive_index, ob_params.material.refractive_index, hitParams.front_face);

            let reflectance = schlick_lazanyi(cos_i, eta_t, 0.0);
            let n_ratio = ray.refractive_index / eta_t;
            let refraction = snell_refract(hitParams.eyev, hitParams.normalv, cos_i, n_ratio);

            next_pcg3d();
            if refraction.w < 0.0 || reflectance >= f32_zero_to_one(rand_pcg3d.y) {
                direction = hitParams.reflectv + (roughness * onb * random_cosine_direction());
                // Reflection stays in the same medium
            } else {
                direction = refraction.xyz;
                p = hitParams.underPoint;
                // Refraction enters the transmitted medium
                next_ri = eta_t;
            }
        }

        let spec_weight = compute_specular_weight(ob_params.material.colour.xyz, ob_params.material.reflective);
        store_ray(offset, Ray(p, next_ri, normalize(direction), ray.pos, ray.throughput * vec4<f32>(spec_weight, 1.0)));
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

            let s_pdf = compute_scattering_pdf(direction, onb[2]);
            var v_light_pdf = 0f;

            for (var i_light: u32 = ubo.lights_offset; i_light < ubo.n_objects; i_light = i_light + 1u) {
                let l_intersection = intersect(next_ray, i_light, true);
                if l_intersection.model_id != i_light || l_intersection.id == -1 || l_intersection.closestT >= MAXLEN {
                    continue;
                }

                v_light_pdf = v_light_pdf + light_pdf(next_ray, l_intersection);
            }

            v_light_pdf = v_light_pdf / f32(ubo.n_objects - ubo.lights_offset);
            let pdf = mis_pdf(s_pdf, v_light_pdf, p_scatter);

            next_ray.throughput = ray.throughput * ob_params.material.colour * s_pdf / pdf;

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

    let img_idx = pixel_pos.y * ubo.resolution.x + pixel_pos.x;
    var color: vec4<f32> = RAY_MISS_COLOUR;
    if ubo.subpixel_idx > 0u {
        color = imageData[img_idx];
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
        imageData[img_idx] = color;
    } else if ubo.subpixel_idx == 0u {
        imageData[img_idx] = color;
    }

    // If ray is still alive after this bounce, write to compaction output
    let updated_throughput = load_ray_throughput(offset);
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
