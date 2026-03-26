#define_import_path rengin::pathtracer_math

#import rengin::constants::{EPSILON, PI}

// Pure math functions extracted from pathtracer.wgsl for testability.
// These have no bind group dependencies — only take parameters and return values.

/// Compute the solid-angle PDF for a sphere light (pbrt 12.2.1).
/// `radius`: bounding radius of the light, `distance`: distance from shading point to light centre.
fn light_pdf_sphere(radius: f32, distance: f32) -> f32 {
    let cos_theta_max = sqrt(1.0 - radius * radius / (distance * distance));
    let solid_angle = 2.0 * PI * (1.0 - cos_theta_max);
    return 1.0 / solid_angle;
}

/// Cosine-weighted hemisphere scattering PDF (pbrt 13.6.3).
fn scattering_pdf(direction: vec3<f32>, normal: vec3<f32>) -> f32 {
    return max(0.0, dot(direction, normal) / PI);
}

/// One-sample MIS balance heuristic (pbrt 13.10.1).
fn mis_pdf_balance(scatter_pdf: f32, light_pdf_val: f32, p_scatter: f32) -> f32 {
    return p_scatter * scatter_pdf + (1.0 - p_scatter) * light_pdf_val;
}

/// One-sample MIS power heuristic β=2 (pbrt 13.10.1).
fn mis_pdf(scatter_pdf: f32, light_pdf_val: f32, p_scatter: f32) -> f32 {
    let ps = p_scatter * scatter_pdf;
    let pl = (1.0 - p_scatter) * light_pdf_val;
    let sum = ps + pl;
    if sum < 1e-10 {
        return 0.0;
    }
    return (ps * ps + pl * pl) / sum;
}

/// Snell's law refraction direction (pbrt 9.3).
/// Returns the refracted direction (unnormalised) and a flag: 1.0 = refracted, -1.0 = TIR.
fn snell_refract(eyev: vec3<f32>, normalv: vec3<f32>, cos_i: f32, n_ratio: f32) -> vec4<f32> {
    let sin_2t = n_ratio * n_ratio * (1.0 - cos_i * cos_i);
    if sin_2t > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0); // TIR
    }
    let cos_t = sqrt(1.0 - sin_2t);
    let direction = normalv * (n_ratio * cos_i - cos_t) - eyev * n_ratio;
    return vec4<f32>(normalize(direction), 1.0);
}

/// Russian roulette termination probability (capped variant, see pathtracer.wgsl).
fn russian_roulette_q(throughput: vec3<f32>) -> f32 {
    let max_component = max(throughput.x, max(throughput.y, throughput.z));
    if max_component >= 1.0 {
        return -1.0; // no RR applied
    }
    return min(0.5, max(0.0, 1.0 - max_component));
}

/// Russian roulette throughput adjustment: divide by survival probability.
fn russian_roulette_adjust(throughput: vec3<f32>, q: f32) -> vec3<f32> {
    return throughput / (1.0 - q);
}

/// Specular weight for mirror / glass materials (pathtracer.wgsl lines 195-198).
fn specular_weight(colour: vec3<f32>, reflective: f32) -> vec3<f32> {
    var w = colour;
    if reflective > 0.0 && reflective < 1.0 {
        w = colour * reflective;
    }
    return w;
}

/// Path regularization: increase roughness after non-specular bounces (pbrt 13.1.4).
fn path_regularization_roughness(reflective: f32, max_component: f32) -> f32 {
    var roughness = 1.0 - reflective;
    if max_component < 0.99 {
        roughness = clamp(roughness * 2.0, 0.1, 0.5);
    }
    return roughness;
}

/// Determine the transmitted-medium refractive index (RI state machine).
fn determine_eta_t(ray_ri: f32, material_ri: f32, front_face: bool) -> f32 {
    var eta_t = material_ri;
    if !front_face {
        if ray_ri > 1.0 + EPSILON {
            eta_t = 1.0;
        }
    }
    return eta_t;
}

/// Cube face sampling: area-weighted face index selection.
/// Returns face index 0..5 for [-x, +x, -y, +y, -z, +z].
fn cube_face_select(scale: vec3<f32>, u: f32) -> u32 {
    let area_yz = scale.y * scale.z;
    let area_xz = scale.x * scale.z;
    let area_xy = scale.x * scale.y;
    let total = 2.0 * (area_yz + area_xz + area_xy);
    let face_rand = u * total;

    if face_rand < area_yz {
        return 0u; // -x
    } else if face_rand < 2.0 * area_yz {
        return 1u; // +x
    } else if face_rand < 2.0 * area_yz + area_xz {
        return 2u; // -y
    } else if face_rand < 2.0 * area_yz + 2.0 * area_xz {
        return 3u; // +y
    } else if face_rand < 2.0 * area_yz + 2.0 * area_xz + area_xy {
        return 4u; // -z
    }
    return 5u; // +z
}
