"""
System statistics helpers
"""
def get_system_info(state, USER_RATING_THRESHOLD, ITEM_RATED_THRESHOLD):
    if state.df is None or state.reduced_df is None or state.user_item_matrix is None:
        return "⚠️ Please initialize the system first!"
    info = []
    info.append("=" * 80)
    info.append("📊 SYSTEM INFORMATION")
    info.append("=" * 80)
    info.append(f"\n🎬 Original Dataset:")
    info.append(f"   • Total ratings: {len(state.df):,}")
    info.append(f"   • Unique users: {state.df['user_id'].nunique():,}")
    info.append(f"   • Unique movies: {state.df['item_id'].nunique():,}")
    from datetime import datetime
    min_ts = state.df['timestamp'].min()
    max_ts = state.df['timestamp'].max()
    try:
        min_date = datetime.fromtimestamp(float(min_ts)).strftime('%Y-%m-%d')
        max_date = datetime.fromtimestamp(float(max_ts)).strftime('%Y-%m-%d')
        info.append(f"   • Date range: {min_date} to {max_date}")
    except Exception:
        info.append(f"   • Date range: {min_ts} to {max_ts}")
    info.append(f"\n🔍 After Filtering (threshold: {USER_RATING_THRESHOLD} ratings/user, {ITEM_RATED_THRESHOLD} ratings/movie):")
    info.append(f"   • Filtered ratings: {len(state.reduced_df):,}")
    info.append(f"   • Active users: {state.reduced_df['user_id'].nunique():,}")
    info.append(f"   • Popular movies: {state.reduced_df['item_id'].nunique():,}")
    info.append(f"\n🔢 User-Item Matrix:")
    info.append(f"   • Dimensions: {state.user_item_matrix.shape[0]:,} users × {state.user_item_matrix.shape[1]:,} movies")
    info.append(f"   • Total cells: {state.user_item_matrix.shape[0] * state.user_item_matrix.shape[1]:,}")
    info.append(f"   • Sparsity: {(1 - state.user_item_matrix.notna().sum().sum() / (state.user_item_matrix.shape[0] * state.user_item_matrix.shape[1])) * 100:.2f}%")
    info.append(f"\n📈 Statistics:")
    info.append(f"   • Average rating: {state.reduced_df['rating'].mean():.2f}")
    info.append(f"   • Median rating: {state.reduced_df['rating'].median():.1f}")
    info.append(f"   • Rating std dev: {state.reduced_df['rating'].std():.2f}")
    info.append(f"   • Data retention: {(len(state.reduced_df) / len(state.df)) * 100:.2f}%")
    info.append("\n" + "=" * 80)
    return "\n".join(info)
