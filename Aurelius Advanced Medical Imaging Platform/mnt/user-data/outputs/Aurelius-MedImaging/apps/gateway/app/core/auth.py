"""Authentication and authorization with Keycloak."""
import jwt
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from keycloak import KeycloakOpenID
from app.core.config import settings

# Initialize HTTP Bearer security
security = HTTPBearer()

# Initialize Keycloak client
keycloak_openid = KeycloakOpenID(
    server_url=settings.KEYCLOAK_URL,
    client_id=settings.KEYCLOAK_CLIENT_ID,
    realm_name=settings.KEYCLOAK_REALM,
    client_secret_key=settings.KEYCLOAK_CLIENT_SECRET,
)


class User:
    """Authenticated user model."""
    
    def __init__(
        self,
        sub: str,
        username: str,
        email: str,
        roles: list[str],
        **kwargs
    ):
        self.sub = sub
        self.username = username
        self.email = email
        self.roles = roles
        self.extra = kwargs
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, roles: list[str]) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Verify JWT token and extract user information.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If token is invalid
    """
    token = credentials.credentials
    
    try:
        # Verify token with Keycloak
        token_info = keycloak_openid.decode_token(
            token,
            key=keycloak_openid.public_key(),
            options={"verify_signature": True, "verify_aud": False, "verify_exp": True}
        )
        
        # Extract user information
        user = User(
            sub=token_info.get("sub"),
            username=token_info.get("preferred_username", ""),
            email=token_info.get("email", ""),
            roles=token_info.get("realm_access", {}).get("roles", []),
            given_name=token_info.get("given_name", ""),
            family_name=token_info.get("family_name", ""),
        )
        
        return user
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: str):
    """
    Dependency to require a specific role.
    
    Args:
        required_role: Role required to access the endpoint
        
    Returns:
        Dependency function
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return user
    
    return role_checker


def require_any_role(required_roles: list[str]):
    """
    Dependency to require any of the specified roles.
    
    Args:
        required_roles: List of roles, any of which grants access
        
    Returns:
        Dependency function
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not user.has_any_role(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {required_roles} required"
            )
        return user
    
    return role_checker


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Optional[User]: Authenticated user or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
