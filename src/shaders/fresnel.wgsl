#define_import_path rengin::fresnel

fn _schlick(cos_i:f32, r0: f32) -> f32 {
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn schlick(cos_i:f32, eta_t: f32, eta_i: f32) -> f32 {
    // // Use Schlick's approximation for reflectance.
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

/// Exact unpolarized dielectric Fresnel reflectance for an interface between
/// `eta_i` and `eta_t` (pbrt 9.5.1).
fn fresnel_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = clamp(abs(cos_theta_i), 0.0, 1.0);
    let sin_2i = max(0.0, 1.0 - cos_i * cos_i);
    let eta = eta_i / eta_t;
    let sin_2t = eta * eta * sin_2i;
    if sin_2t >= 1.0 {
        return 1.0;
    }

    let cos_t = sqrt(max(0.0, 1.0 - sin_2t));
    let r_parallel = (eta_t * cos_i - eta_i * cos_t) /
        (eta_t * cos_i + eta_i * cos_t);
    let r_perpendicular = (eta_i * cos_i - eta_t * cos_t) /
        (eta_i * cos_i + eta_t * cos_t);
    return 0.5 * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

fn _schlick_lazanyi(cos_i:f32, eta_t: f32, eta_i: f32, a: f32, alpha:f32) -> f32 {
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return _schlick(cos_i, r0) - a * cos_i * pow(1.0 - cos_i , alpha);
}

fn schlick_lazanyi(cos_i:f32, eta_t: f32, k: f32) -> f32 {
    return (pow(eta_t - 1.0, 2.0) + 4.0 * eta_t * pow(1.0 - cos_i,5.0) + pow(k,2.0)) / (pow(eta_t+1.0,2.0) + pow(k,2.0));
}
