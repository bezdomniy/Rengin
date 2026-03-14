/// Refraction logic mirroring the WGSL pathtracer shader.
/// This module exists solely for unit testing the RI tracking decisions.

const EPSILON: f32 = 0.001;

/// Result of a refraction decision at a dielectric surface.
#[derive(Debug, PartialEq)]
pub enum RefractionResult {
    /// Ray refracts into the transmitted medium.
    Refract {
        n_ratio: f32,
        next_ri: f32,
    },
    /// Total internal reflection — ray stays in the same medium.
    TotalInternalReflection {
        next_ri: f32,
    },
}

/// Compute the refraction parameters at a dielectric boundary.
///
/// This mirrors the logic in pathtracer.wgsl for the transparency path,
/// excluding the Fresnel random reflection (only tests the deterministic parts).
///
/// # Arguments
/// * `ray_ri` - Current refractive index of the ray's medium
/// * `material_ri` - Refractive index of the hit material
/// * `front_face` - Whether the ray hit the front face (entering) or back face (exiting)
/// * `cos_i` - Cosine of the incident angle (dot of eye vector and normal)
pub fn compute_refraction(
    ray_ri: f32,
    material_ri: f32,
    front_face: bool,
    cos_i: f32,
) -> RefractionResult {
    // Determine the transmitted medium's RI (mirrors shader logic)
    let eta_t = if front_face {
        material_ri
    } else if ray_ri > 1.0 + EPSILON {
        // Ray is inside a dense medium (glass) → exiting to air
        1.0
    } else {
        // Ray is in air hitting back face → entering this material
        material_ri
    };

    let n_ratio = ray_ri / eta_t;
    let sin_2t = n_ratio * n_ratio * (1.0 - cos_i * cos_i);

    if sin_2t > 1.0 {
        // Total internal reflection
        RefractionResult::TotalInternalReflection { next_ri: ray_ri }
    } else {
        RefractionResult::Refract {
            n_ratio,
            next_ri: eta_t,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: f32 = 1.0;
    const GLASS: f32 = 1.52;
    const DIAMOND: f32 = 2.42;
    const COS_NORMAL: f32 = 1.0; // Head-on incidence
    const COS_45: f32 = 0.7071; // 45 degree incidence

    fn assert_approx(a: f32, b: f32, label: &str) {
        assert!(
            (a - b).abs() < 0.01,
            "{}: expected {:.4}, got {:.4}",
            label,
            b,
            a
        );
    }

    // ================================================================
    // Single glass sphere: air → glass → air
    // ================================================================

    #[test]
    fn single_glass_sphere_enter() {
        // Ray in air hits front face of glass sphere
        let result = compute_refraction(AIR, GLASS, true, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, AIR / GLASS, "n_ratio entering glass");
                assert_approx(next_ri, GLASS, "next_ri should be glass");
            }
            _ => panic!("Should refract, not TIR"),
        }
    }

    #[test]
    fn single_glass_sphere_exit() {
        // Ray in glass hits back face of glass sphere
        let result = compute_refraction(GLASS, GLASS, false, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, GLASS / AIR, "n_ratio exiting glass");
                assert_approx(next_ri, AIR, "next_ri should be air");
            }
            _ => panic!("Should refract at normal incidence, not TIR"),
        }
    }

    #[test]
    fn single_glass_sphere_roundtrip_ri() {
        // Full path: air → glass → air. Final RI should be 1.0.
        let r1 = compute_refraction(AIR, GLASS, true, COS_NORMAL);
        let ri_after_enter = match r1 {
            RefractionResult::Refract { next_ri, .. } => next_ri,
            _ => panic!("Should refract"),
        };
        assert_approx(ri_after_enter, GLASS, "after entering glass");

        let r2 = compute_refraction(ri_after_enter, GLASS, false, COS_NORMAL);
        let ri_after_exit = match r2 {
            RefractionResult::Refract { next_ri, .. } => next_ri,
            _ => panic!("Should refract"),
        };
        assert_approx(ri_after_exit, AIR, "after exiting glass back to air");
    }

    // ================================================================
    // Hollow glass sphere: air → glass → air → glass → air
    // Outer sphere: RI=1.52, Inner sphere: RI=1.0
    // ================================================================

    #[test]
    fn hollow_glass_enter_outer() {
        // Step 1: air → glass (outer sphere front face)
        let result = compute_refraction(AIR, GLASS, true, COS_NORMAL);
        match result {
            RefractionResult::Refract { next_ri, .. } => {
                assert_approx(next_ri, GLASS, "should enter glass");
            }
            _ => panic!("Should refract"),
        }
    }

    #[test]
    fn hollow_glass_enter_air_pocket() {
        // Step 2: glass → air (inner sphere front face, material RI=1.0)
        let result = compute_refraction(GLASS, AIR, true, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, GLASS / AIR, "n_ratio glass to air");
                assert_approx(next_ri, AIR, "should enter air pocket");
            }
            _ => panic!("Should refract"),
        }
    }

    #[test]
    fn hollow_glass_exit_air_pocket() {
        // Step 3: air → glass (inner sphere back face, material RI=1.0)
        // Ray is in air (RI=1.0), hitting back face of inner sphere (material RI=1.0).
        // Heuristic: ray_ri <= 1.0+eps → eta_t = material_ri = 1.0
        // This means n_ratio = 1.0/1.0 = 1.0 → no refraction.
        // KNOWN LIMITATION: should re-enter glass (1.52) but we don't know the outer medium.
        let result = compute_refraction(AIR, AIR, false, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                // With current heuristic: no refraction at this boundary
                assert_approx(n_ratio, 1.0, "n_ratio (known limitation: should be 1.0/1.52)");
                assert_approx(next_ri, AIR, "next_ri (known limitation: should be 1.52)");
            }
            _ => panic!("Should not TIR at normal incidence"),
        }
    }

    #[test]
    fn hollow_glass_exit_outer() {
        // Step 4: after the air pocket, ray hits outer sphere back face.
        // Due to limitation in step 3, ray is still at RI=1.0.
        // Back face of outer sphere (material RI=1.52), ray RI=1.0:
        // ray_ri <= 1.0+eps → eta_t = material_ri = 1.52 → entering glass
        let result = compute_refraction(AIR, GLASS, false, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, AIR / GLASS, "n_ratio air to glass");
                assert_approx(next_ri, GLASS, "enters glass (known limitation: should exit to air)");
            }
            _ => panic!("Should refract"),
        }
    }

    // ================================================================
    // Diamond sphere (high RI)
    // ================================================================

    #[test]
    fn diamond_sphere_enter() {
        let result = compute_refraction(AIR, DIAMOND, true, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, AIR / DIAMOND, "n_ratio entering diamond");
                assert_approx(next_ri, DIAMOND, "should enter diamond");
            }
            _ => panic!("Should refract"),
        }
    }

    #[test]
    fn diamond_sphere_exit() {
        let result = compute_refraction(DIAMOND, DIAMOND, false, COS_NORMAL);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, DIAMOND / AIR, "n_ratio exiting diamond");
                assert_approx(next_ri, AIR, "should exit to air");
            }
            _ => panic!("Should refract at normal incidence"),
        }
    }

    #[test]
    fn diamond_total_internal_reflection() {
        // Diamond has critical angle ~24.4°, so cos(24.4°) ≈ 0.91
        // At 45° incidence from inside diamond: cos_i = 0.707 < 0.91 → TIR
        let result = compute_refraction(DIAMOND, DIAMOND, false, COS_45);
        match result {
            RefractionResult::TotalInternalReflection { next_ri } => {
                assert_approx(next_ri, DIAMOND, "TIR keeps ray in diamond");
            }
            _ => panic!("Should be TIR at 45° inside diamond"),
        }
    }

    #[test]
    fn glass_no_tir_at_normal_incidence() {
        // Normal incidence should never produce TIR
        let result = compute_refraction(GLASS, GLASS, false, COS_NORMAL);
        match result {
            RefractionResult::Refract { .. } => {} // expected
            _ => panic!("Normal incidence should not produce TIR"),
        }
    }

    // ================================================================
    // Reflection-only materials (no transparency)
    // ================================================================

    #[test]
    fn reflection_preserves_ri() {
        // When a ray reflects (not refracts), it stays in the same medium.
        // This is tested by checking that next_ri = ray_ri for TIR.
        let result = compute_refraction(DIAMOND, DIAMOND, false, COS_45);
        match result {
            RefractionResult::TotalInternalReflection { next_ri } => {
                assert_approx(next_ri, DIAMOND, "reflection stays in same medium");
            }
            _ => panic!("Expected TIR"),
        }
    }

    // ================================================================
    // Edge cases
    // ================================================================

    #[test]
    fn air_to_air_no_refraction() {
        // RI=1.0 on both sides → no bending
        let result = compute_refraction(AIR, AIR, true, COS_45);
        match result {
            RefractionResult::Refract { n_ratio, next_ri } => {
                assert_approx(n_ratio, 1.0, "air to air: no bending");
                assert_approx(next_ri, AIR, "stays in air");
            }
            _ => panic!("Should not TIR"),
        }
    }

    #[test]
    fn grazing_angle_glass_exit() {
        // Near-grazing angle from inside glass (cos_i very small)
        let cos_grazing = 0.1;
        let result = compute_refraction(GLASS, GLASS, false, cos_grazing);
        // sin_2t = (1.52/1.0)^2 * (1 - 0.01) = 2.3104 * 0.99 = 2.287 > 1 → TIR
        match result {
            RefractionResult::TotalInternalReflection { next_ri } => {
                assert_approx(next_ri, GLASS, "TIR at grazing angle");
            }
            _ => panic!("Should be TIR at grazing angle inside glass"),
        }
    }
}
