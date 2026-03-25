#define_import_path rengin::fresnel

fn _schlick(cos_i:f32, r0: f32) -> f32 {
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn schlick(cos_i:f32, eta_t: f32, eta_i: f32) -> f32 {
    // // Use Schlick's approximation for reflectance.
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn _schlick_lazanyi(cos_i:f32, eta_t: f32, eta_i: f32, a: f32, alpha:f32) -> f32 {
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return _schlick(cos_i, r0) - a * cos_i * pow(1.0 - cos_i , alpha);
}

fn schlick_lazanyi(cos_i:f32, eta_t: f32, k: f32) -> f32 {
    return (pow(eta_t - 1.0, 2.0) + 4.0 * eta_t * pow(1.0 - cos_i,5.0) + pow(k,2.0)) / (pow(eta_t+1.0,2.0) + pow(k,2.0));
}
